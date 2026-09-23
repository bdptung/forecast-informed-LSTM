

import os
import torch
import torch.nn as nn


import time


class AttrGRU(nn.Module):
    def __init__(self, input_dim, attr_size, hidden_dim=128, num_layers=2, out_len=168, dropout=0.2):
        """
        Args:
            input_dim:   number of input features per time step (e.g., temp, masks, time feats…)
            hidden_dim:  GRU hidden size
            num_layers:  GRU depth
            output_len:  forecast horizon (number of steps to predict)
            output_dim:  number of outputs per step (e.g., 1 for temperature)
            dropout:     dropout between GRU layers (ignored if num_layers=1)
        """
        super().__init__()
        self.model_type = 'standard'
        self.output_len = out_len
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, out_len * attr_size)
        self.attr_size = attr_size

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, output_len, attr_size)
        """
        _, h = self.gru(x)                    # h: (num_layers, batch, hidden_dim)
        h_last = h[-1]                        # (batch, hidden_dim)
        out = self.fc(h_last)                 # (batch, output_len*output_dim)
        out = out.view(-1, self.output_len, self.attr_size)
        return out

class AttrSeq2SeqGRU(nn.Module):
    def __init__(self, enc_in_dim, dec_tf_dim, attr_size,
                 hidden_dim=128, layers=2, out_len=168, dropout=0.2):
        """
        enc_in_dim: past features (time feats + attr observed + attr) -> 6 + s + s
        dec_tf_dim: future time features only = 6
        return_sequence: if True -> (B, 168, 1), else -> (B, 168)
        """
        super().__init__()
        self.model_type = 'seq2seq'
        print(f"ENCODER DIM: {enc_in_dim}; DECODER DIM: {dec_tf_dim}")
        self.out_len = out_len
        self.attr_size = attr_size

        # self.id_emb = nn.Embedding(n_stations, id_emb_dim)
        self.encoder = nn.GRU(enc_in_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        # decoder input at each step: previous y_hat (1) + future time feats (4) = 5
        self.decoder = nn.GRU(attr_size + dec_tf_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        # self.proj = nn.Linear(hidden_dim, attr_size)  # predict normalized attribute
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, attr_size),
        )

        self.forecast_proj = nn.Linear(forecast_dim, hidden_dim) if forecast_dim is not None else None

    def forward(self, enc_seq, dec_time_feats, teacher_forcing_ratio=0.0, y_future=None, y_mask=None):
        """
        enc_seq: (B, 1440, 6+s+s)
        dec_time_feats: (B, 168, 6)
        y_future: (B, 168, s) normalized, optional for teacher forcing
        returns: (B, 168, s)
        """

        B = enc_seq.size(0)

        # Encode
        _, h = self.encoder(enc_seq)  # (layers, B, hidden)

        # Init dec input with last known values = last values in encoder sequence
        last_known_values = enc_seq[:, -1:, -self.attr_size:]  # (B,s)
        outputs = []
        dec_h = h
        dec_in = torch.cat([last_known_values, dec_time_feats[:, 0:1, :]], dim=-1)  # (B,1,6)
        for t in range(self.out_len):
            out, dec_h = self.decoder(dec_in, dec_h)  # out: (B,1,H)
            y_hat = self.proj(out)  # (B,1,s)
            outputs.append(y_hat)

            # next decoder input: use teacher forcing sometimes
            if (self.training and y_future is not None and torch.rand(1).item() < teacher_forcing_ratio):
                # if teacher forcing and has nan, use prediction instead
                if y_mask[:, t].sum() == 0:
                    prev = y_hat
                else:
                    prev = y_future[:, t:t+1, :]  # (B,1,s)
            else:
                prev = y_hat  # (B,1,s)

            if t+1 < self.out_len:
                tf_next = dec_time_feats[:, t+1:t+2, :]  # (B,1,4)
                dec_in = torch.cat([prev, tf_next], dim=-1)  # (B,1,5)

        y_seq = torch.cat(outputs, dim=1)  # (B,168,s)
        return y_seq

class AttrLSTM(nn.Module):
    """Plain LSTM baseline (no encoder-decoder, non-autoregressive), but conditioned on the same
    information as the seq2seq models: it encodes the input to a single context vector, then a
    per-step MLP head predicts every horizon step from [context, future time feats, per-day forecast].
    Forecast handling (per-day alignment, has-forecast flag, lead-time) matches AttrSeq2SeqLSTM_v1."""
    def __init__(self, enc_in_dim, dec_tf_dim, attr_size,
                 hidden_dim=128, layers=2, out_len=168, dropout=0.2,
                 forecast_dim=None, forecast_days=4):
        super().__init__()
        self.model_type = 'plain_lstm'
        self.out_len = out_len
        self.attr_size = attr_size
        self.forecast_dim = forecast_dim               # per-day forecast feature dim
        self.forecast_days = forecast_days
        _fct_step_dim = (forecast_dim + 2) if forecast_dim is not None else 0

        print(f"ENCODER DIM: {enc_in_dim}; HEAD DIM: {hidden_dim + dec_tf_dim + _fct_step_dim}")

        self.lstm = nn.LSTM(enc_in_dim, hidden_dim, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        # full multi-day forecast (flattened) -> global context
        self.forecast_proj = nn.Linear(forecast_dim * forecast_days, hidden_dim) if self.forecast_dim is not None else None
        # per-step head: global context + future time feats + relevant-day forecast slice -> attr
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + dec_tf_dim + _fct_step_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, attr_size),
        )

    def forward(self, enc_seq, dec_time_feats, last_val=None, station_feats=None, forecast=None):
        """
        enc_seq: (B, T, enc_in_dim); dec_time_feats: (B, out_len, dec_tf_dim)
        returns: (B, out_len, attr_size)
        """
        B = enc_seq.size(0)
        _, (h, c) = self.lstm(enc_seq)
        ctx = h[-1]                                    # (B, H) global context from the input

        if self.forecast_dim is not None and forecast is not None:
            # forecast: (B, n_days, day_dim); day d is the forecast for (current_day + 1 + d)
            n_days = forecast.size(1)
            day_dim = forecast.size(2)
            ctx = ctx + self.forecast_proj(forecast.reshape(B, n_days * day_dim))
            # per-step forecast day aligned via midnight detection (minute-of-day cols 0,1 == (0,1))
            is_midnight = (dec_time_feats[:, :, 0] == 0) & (dec_time_feats[:, :, 1] == 1)   # (B, out_len)
            day_ahead = is_midnight.to(forecast.dtype).cumsum(dim=1)
            fday = day_ahead - 1.0                                                          # day_ahead 1 -> forecast day 0
            valid = (fday >= 0) & (fday <= (n_days - 1))
            gather_idx = fday.clamp(0, n_days - 1).long().unsqueeze(-1).expand(B, self.out_len, day_dim)
            per_step = torch.gather(forecast, 1, gather_idx)                                # (B, out_len, day_dim)
            flag = valid.to(forecast.dtype).unsqueeze(-1)                                   # (B, out_len, 1)
            per_step = per_step * flag
            lead = (day_ahead / (self.out_len / 24.0)).unsqueeze(-1)                        # (B, out_len, 1)
            per_step = torch.cat([per_step, flag, lead], dim=-1)                            # (B, out_len, day_dim+2)
            feats = torch.cat([ctx.unsqueeze(1).expand(B, self.out_len, -1), dec_time_feats, per_step], dim=-1)
        else:
            feats = torch.cat([ctx.unsqueeze(1).expand(B, self.out_len, -1), dec_time_feats], dim=-1)

        return self.head(feats)  # (B, out_len, attr_size)

class AttrLSTMv1(nn.Module):
    def __init__(self, input_dim, dec_tf_dim, attr_size,
                 hidden_dim=256, num_layers=2, out_len=168, dropout=0.2, station_dim=3):
        super().__init__()
        self.model_type = 'standard_w_context'
        self.output_len = out_len
        self.attr_size = attr_size
        self.enc_lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.context_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.station_proj = nn.Linear(station_dim, hidden_dim)
        # self.ctx_c_proj   = nn.Linear(hidden_dim, hidden_dim)
        # non-autoregressive decoder: single LSTM pass over all output steps
        # input: future time features + last observed values (fixed, not fed-back predictions)
        # context seeds the hidden state so each step gets a different hidden state
        self.dec_lstm = nn.LSTM(dec_tf_dim + attr_size, hidden_dim, 1,
                                batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attr_size),
        )

    def forward(self, x, dec_time_feats, station_feats=None):
        enc_out, (h, _) = self.enc_lstm(x)
        h_last     = h[-1]                              # (B, H)
        enc_mean   = enc_out.mean(dim=1)                # (B, H)
        enc_max    = enc_out.max(dim=1).values          # (B, H)
        enc_recent = enc_out[:, -24:, :].mean(dim=1)    # (B, H)
        context    = self.context_proj(
            torch.cat([h_last, enc_mean, enc_max, enc_recent], dim=-1)
        )                                               # (B, H)

        if station_feats is not None:
            context = context + self.station_proj(station_feats)  # (B, H)

        # last observed output values — OUT_ATTRS must be last attr_size cols of x
        last_obs        = x[:, -1, -self.attr_size:]                               # (B, attr_size)
        last_obs_expand = last_obs.unsqueeze(1).expand(-1, self.output_len, -1)

        # single LSTM pass: hidden state evolves at each step, reducing uniform smoothness
        dec_input = torch.cat([dec_time_feats, last_obs_expand], dim=-1)    # (B, 168, tf+attr)
        ctx_h = context.unsqueeze(0).contiguous()
        ctx_c = self.ctx_c_proj(context).unsqueeze(0).contiguous()
        dec_out, _ = self.dec_lstm(dec_input, (ctx_h, torch.zeros_like(ctx_h)))

        return self.proj(dec_out)                                           # (B, 168, attr_size)

class AttrSeq2SeqLSTM(nn.Module):
    def __init__(self, enc_in_dim, dec_tf_dim, attr_size,
                 hidden_dim=128, layers=2, out_len=168, dropout=0.2):
        """
        enc_in_dim: past features (time feats + attr observed + attr) -> 6 + s + s
        dec_tf_dim: future time features only = 6
        return_sequence: if True -> (B, 168, 1), else -> (B, 168)
        """
        super().__init__()
        self.model_type = 'seq2seq'
        print(f"ENCODER DIM: {enc_in_dim}; DECODER DIM: {dec_tf_dim}")
        self.out_len = out_len
        self.attr_size = attr_size

        # self.id_emb = nn.Embedding(n_stations, id_emb_dim)
        self.encoder = nn.LSTM(enc_in_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        # decoder input at each step: previous y_hat (1) + future time feats (4) = 5
        self.decoder = nn.LSTM(attr_size + dec_tf_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        # self.proj = nn.Linear(hidden_dim, attr_size)  # predict normalized attribute
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, attr_size),
        )

    def forward(self, enc_seq, dec_time_feats, teacher_forcing_ratio=0.0, y_future=None, y_mask=None):
        """
        enc_seq: (B, 1440, 6+s+s)
        dec_time_feats: (B, 168, 6)
        y_future: (B, 168, s) normalized, optional for teacher forcing
        returns: (B, 168, s)
        """

        B = enc_seq.size(0)

        # Encode
        _, (h, c) = self.encoder(enc_seq)  # (layers, B, hidden)

        # Init dec input with last known values = last values in encoder sequence
        last_known_values = enc_seq[:, -1:, -self.attr_size:]  # (B,s)
        outputs = []
        dec_h, dec_c = h, c
        dec_in = torch.cat([last_known_values, dec_time_feats[:, 0:1, :]], dim=-1)  # (B,1,6)
        for t in range(self.out_len):
            out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))  # out: (B,1,H)
            y_hat = self.proj(out)  # (B,1,s)
            outputs.append(y_hat)

            # next decoder input: use teacher forcing sometimes
            if (self.training and y_future is not None and torch.rand(1).item() < teacher_forcing_ratio):
                # if teacher forcing and has nan, use prediction instead
                if y_mask[:, t].sum() == 0:
                    prev = y_hat
                else:
                    prev = y_future[:, t:t+1, :]  # (B,1,s)
            else:
                prev = y_hat  # (B,1,s)

            if t+1 < self.out_len:
                tf_next = dec_time_feats[:, t+1:t+2, :]  # (B,1,4)
                dec_in = torch.cat([prev, tf_next], dim=-1)  # (B,1,5)

        y_seq = torch.cat(outputs, dim=1)  # (B,168,s)
        return y_seq



# def train_one_epoch(model, loader, optim, criterion, clip_grad=1.0, tf_ratio=0.5):
#     model.train()
#     total = 0.0
#     times = []
#     for i in range(5):
#         times.append(torch.cuda.Event(enable_timing=True))
#     for enc_seq, dec_tf, mask, y in loader:
#         i0 = 0
#         times[i0].record()
#         pred = model_run(model, enc_seq, dec_tf, teacher_forcing_ratio=tf_ratio, y_future=y, y_mask=mask)

#         i0 +=1
#         times[i0].record()
        
#         loss = criterion(pred, y, mask)
#         optim.zero_grad()

#         i0 +=1
#         times[i0].record()

#         loss.backward()

#         i0 +=1
#         times[i0].record()
    
#         if clip_grad is not None:
#             norm = torch.nn.utils.get_total_norm(model.parameters())
#             torch.nn.utils.clip_grads_with_norm_(model.parameters(), clip_grad, norm, True)
#             # norm = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#             print(f'      norm: {norm.item():.2f}; clip_grad: {clip_grad:2f}')
#         optim.step()

#         i0 +=1
#         times[i0].record()

#         total += loss.item() * enc_seq.size(0)
#         print('      train batch:', end=' ')
#         for i in range(1, len(times)):
#             print(f'{(times[i-1].elapsed_time(times[i])):1f}', end=' ')
#         print('\n')
#         i0 = 0
#         times[i0].record()

#     return total / len(loader.dataset)

def train_one_epoch(model, loader, optim, criterion, clip_grad=1.0, tf_ratio=0.5, scheduler=None):
    model.train()
    total = 0.0
    for enc_seq, dec_tf, mask, y in loader:
        pred = model_run(model, enc_seq, dec_tf, teacher_forcing_ratio=tf_ratio, y_future=y, y_mask=mask)
        loss = criterion(pred, y, mask)
        optim.zero_grad()
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            # norm = torch.nn.utils.get_total_norm(model.parameters())
            # torch.nn.utils.clip_grads_with_norm_(model.parameters(), clip_grad, norm, True)
            # print(f'      norm: {norm.item():.2f}')
        optim.step()
        if scheduler: scheduler.step()
        total += loss.item() * enc_seq.size(0)
    return total / len(loader.dataset)

def train_one_epoch_scaler(model, loader, optimizer, criterion, scaler, clip_grad=1.0, tf_ratio=0.5, scheduler=None, station_feats=None):
    model.train()
    total_loss = 0.0
    for enc_seq, dec_tf, mask, y in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            pred = model_run(model, enc_seq, dec_tf, teacher_forcing_ratio=tf_ratio, y_future=y, y_mask=mask)
            loss = criterion(pred, y, mask)
        if scaler:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()
        total_loss += loss.item() * enc_seq.size(0)
    return total_loss / len(loader.dataset)

def train_one_epoch_scaler_v1(model, loader, optimizer, criterion, scaler, clip_grad=1.0, tf_ratio=0.0, scheduler=None, epoch=None):
    model.train()
    total_loss = 0.0
    count = 0
    count_total = 0
    start = time.perf_counter()
    start0 = time.perf_counter()
    for enc_seq, dec_tf, mask, y, stn, last_val, forecast in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            pred = model_run(model, enc_seq, dec_tf, last_val=last_val, station_feats=stn, forecast=forecast)
            loss = criterion(pred, y, mask)
        if scaler:
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()
        total_loss += loss.item() * enc_seq.size(0)
        count += enc_seq.size(0)
        if count > 5000:
            count_total += count
            end = time.perf_counter()
            print(f'      {f'Epoch {epoch} - ' if epoch is not None else ""}train - cumulative loss={(total_loss / count_total):#.6f}, time elapsed={end-start:.2f}s, {count_total:,}/{len(loader.dataset):,}')
            start = end
            count = 0
    print(f'      {f'Epoch {epoch} - ' if epoch is not None else ""}train - cumulative loss={(total_loss / len(loader.dataset)):#.6f} , time elapsed={time.perf_counter()-start0:.2f}s')
    return total_loss / len(loader.dataset)

# def train_one_epoch_scaler(model, loader, optimizer, criterion, scaler, clip_grad=1.0, tf_ratio=0.5, accu=1):
#     model.train()
#     total_loss = 0.0
#     ac = accu
#     optimizer.zero_grad(set_to_none=True)
#     for enc_seq, dec_tf, mask, y in loader:
#         with torch.amp.autocast('cuda', dtype=torch.float16):
#             pred = model_run(model, enc_seq, dec_tf, teacher_forcing_ratio=tf_ratio, y_future=y, y_mask=mask)
#             loss = criterion(pred, y, mask)
#         scaled_loss = scaler.scale(loss)
#         scaled_loss.backward()

#         total_loss += scaled_loss.item() * enc_seq.size(0)
        
#         ac -= 1
#         if ac == 0:
#             scaler.unscale_(optimizer)
#             if clip_grad is not None:
#                 nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#             scaler.step(optimizer); scaler.update()
#             optimizer.zero_grad(set_to_none=True)
#             ac = accu
#         # optim.zero_grad()
#         # loss.backward()
#         # if clip_grad is not None:
#         #     nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#         #     # print(f'      norm: {norm.item():.2f}')
#         # optim.step()
#     if ac != accu:
#         scaler.unscale_(optimizer)
#         if clip_grad is not None:
#             nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#         scaler.step(optimizer); scaler.update()
#         optimizer.zero_grad(set_to_none=True)

#     return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, station_feats=None):
    model.eval()
    total = 0.0
    for enc_seq, dec_tf, mask, y in loader:
        stn = station_feats.expand(enc_seq.size(0), -1) if station_feats is not None else None
        pred = model_run(model, enc_seq, dec_tf, 0.0, station_feats=stn)
        loss = criterion(pred, y, mask)
        total += loss.item() * enc_seq.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate_v1(model, loader, criterion, epoch=None):
    model.eval()
    total_loss = 0.0
    count = 0
    count_total = 0
    start = time.time()
    for enc_seq, dec_tf, mask, y, stn, last_val, forecast in loader:
        with torch.autocast('cuda'):
            pred = model_run(model, enc_seq, dec_tf, 0.0, last_val=last_val, station_feats=stn, forecast=forecast)
            loss = criterion(pred, y, mask, with_grad_penalty=False)  # eval = pure prediction error, no shape term
        total_loss += loss.item() * enc_seq.size(0)
        count += enc_seq.size(0)
        if count > 5000:
            count_total += count
            end = time.time()
            print(f'      {f'Epoch {epoch} - ' if epoch is not None else ""}test - cumulative loss={(total_loss / count_total):#.6f}, time elapsed={end-start:.2f}s, {count_total:,}/{len(loader.dataset):,}')
            start = end
            count = 0
    print(f'      {f'Epoch {epoch} - ' if epoch is not None else ""}test - cumulative loss={(total_loss / len(loader.dataset)):#.6f} , time elapsed={time.time()-start:.2f}s')
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_scaler(model, loader, criterion, scaler):
    model.eval()
    total = 0.0
    for enc_seq, dec_tf, mask, y in loader:
        with torch.autocast('cuda'):
            pred = model_run(model, enc_seq, dec_tf, 0.0)
            loss = criterion(pred, y, mask)
        # scaled_loss = scaler.scale(loss)
        # total += scaled_loss.item() * enc_seq.size(0)
        total += loss.item() * enc_seq.size(0)   # remove scaler.scale()
    return total / len(loader.dataset)

def model_run(model, enc_seq, dec_tf=None, teacher_forcing_ratio=0.0, y_future=None, y_mask=None, last_val=None, station_feats=None, forecast=None):
    if model.model_type == 'seq2seq':
        return model(enc_seq, dec_tf, teacher_forcing_ratio, y_future=y_future, y_mask=y_mask)
    elif model.model_type == 'seq2seq_v1':
        return model(enc_seq, dec_tf, last_val=last_val, station_feats=station_feats, forecast=forecast)
    elif model.model_type == 'seq2seq_cnn':
        return model(enc_seq, dec_tf, last_val=last_val, station_feats=station_feats, forecast=forecast)
    elif model.model_type == 'seq2seq_cnn_attn':
        return model(enc_seq, dec_tf, last_val=last_val, station_feats=station_feats, forecast=forecast)
    elif model.model_type == 'seq2seq_attn':
        return model(enc_seq, dec_tf, last_val=last_val, station_feats=station_feats, forecast=forecast)
    elif model.model_type == 'seq2seq_bilstm':
        return model(enc_seq, dec_tf, last_val=last_val, station_feats=station_feats, forecast=forecast)
    elif model.model_type == 'standard_w_context':
        return model(enc_seq, dec_tf, station_feats=station_feats)
    elif model.model_type == 'plain_lstm':
        return model(enc_seq, dec_tf, last_val=last_val, station_feats=station_feats, forecast=forecast)
    else:
        return model(enc_seq)


class AttrSeq2SeqLSTM_v1(nn.Module):
    def __init__(self, enc_in_dim, dec_tf_dim, attr_size,
                 hidden_dim=128, layers=2, out_len=168, dropout=0.2,
                 station_dim=3, forecast_dim=None, forecast_days=4):
        """
        enc_in_dim: past features (time feats + attr observed + attr) -> 6 + s + s
        dec_tf_dim: future time features only = 6
        forecast_dim: per-DAY forecast feature dim (e.g. 10); the forecast tensor is (B, forecast_days, forecast_dim)
        forecast_days: number of forecast days available (e.g. 4); horizon steps beyond this get a zeroed slice + flag=0
        return_sequence: if True -> (B, 168, 1), else -> (B, 168)
        """
        super().__init__()
        self.model_type = 'seq2seq_v1'
        self.out_len = out_len
        self.attr_size = attr_size

        # self.station_dim = station_dim
        self.forecast_dim = forecast_dim          # per-day forecast feature dim
        self.forecast_days = forecast_days
        # per decoder step we append the relevant day's forecast slice + has-forecast flag + lead-time (+2)
        _fct_step_dim = (forecast_dim + 2) if forecast_dim is not None else 0

        print(f"ENCODER DIM: {enc_in_dim}; DECODER DIM: {attr_size + dec_tf_dim + _fct_step_dim}")

        # self.id_emb = nn.Embedding(n_stations, id_emb_dim)
        self.encoder = nn.LSTM(enc_in_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        # decoder input at each step: previous y_hat (attr) + future time feats + relevant-day forecast slice + flag
        self.decoder = nn.LSTM(attr_size + dec_tf_dim + _fct_step_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, attr_size),
        )
        # full multi-day forecast (flattened) -> initial decoder state for global multi-day context
        self.forecast_proj = nn.Linear(forecast_dim * forecast_days, hidden_dim) if self.forecast_dim is not None else None


    def forward(self, enc_seq, dec_time_feats, last_val=None, station_feats=None, forecast=None):
        """
        enc_seq: (B, 1440, 6+s+s)
        dec_time_feats: (B, 168, 6)
        y_future: (B, 168, s) normalized, optional for teacher forcing
        returns: (B, 168, s)
        """

        B = enc_seq.size(0)

        # Encode
        _, (h, c) = self.encoder(enc_seq)  # (layers, B, hidden)

        if self.forecast_dim is not None and forecast is not None:
            # forecast: (B, n_days, day_dim); day d (0-indexed) is the forecast for (current_day + 1 + d),
            # i.e. the forecast does NOT cover the current day.
            n_days = forecast.size(1)
            day_dim = forecast.size(2)
            # global context: full multi-day forecast (flattened) into the initial decoder state
            f = self.forecast_proj(forecast.reshape(B, n_days * day_dim))
            h = h + f.unsqueeze(0)
            c = c + f.unsqueeze(0)

            # Align each horizon step to the right forecast day. A new calendar day is the midnight step,
            # detected directly from minute-of-day (cols 0,1 = sin,cos): at 00:00 sin==0 and cos==1.
            # Counting midnights from step 0 gives day_ahead (no input-end anchor needed);
            # day_ahead==1 -> forecast day 0 (forecast excludes current day). This also handles the
            # year/leap boundary that a day-of-year change check would miss.
            is_midnight = (dec_time_feats[:, :, 0] == 0) & (dec_time_feats[:, :, 1] == 1)   # (B, out_len)
            day_ahead = is_midnight.to(forecast.dtype).cumsum(dim=1)                        # midnights crossed
            fday = day_ahead - 1.0                                            # day_ahead 1 -> forecast day 0
            valid = (fday >= 0) & (fday <= (n_days - 1))                                               # (B, out_len)
            gather_idx = fday.clamp(0, n_days - 1).long().unsqueeze(-1).expand(B, self.out_len, day_dim)
            per_step = torch.gather(forecast, 1, gather_idx)                                           # (B, out_len, day_dim)
            flag = valid.to(forecast.dtype).unsqueeze(-1)                                              # (B, out_len, 1)
            per_step = per_step * flag                                                                 # zero out non-covered days
            # normalized forecast lead time (days ahead of the last observation): lets the model tell
            # near-term (rest of today, day_ahead 0) from far future (days 5-7), which both have flag=0.
            lead = (day_ahead / (self.out_len / 24.0)).unsqueeze(-1)                                   # (B, out_len, 1)
            per_step = torch.cat([per_step, flag, lead], dim=-1)                                       # (B, out_len, day_dim+2)
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :], per_step[:, t:t+1, :]]
            # print('prev_doy', prev_doy[0])
            # print('day_changed', day_changed[0])
            # print('day_ahead', day_ahead[0])
            # print('per_step', per_step[0])
            # print('flag', flag[0])
        else:
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :]]

        # Init dec input with last known values = last values in encoder sequence
        last_known_values = last_val if last_val is not None else enc_seq[:, -1:, -self.attr_size:]
        outputs = []
        dec_h, dec_c = h, c
        dec_in = torch.cat(dec_parts(last_known_values, 0), dim=-1)

        for t in range(self.out_len):
            out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))  # out: (B,1,H)
            y_hat = self.proj(out)  # (B,1,s)
            outputs.append(y_hat)

            prev = y_hat  # (B,1,s)

            if t+1 < self.out_len:
                dec_in = torch.cat(dec_parts(prev, t+1), dim=-1)

        y_seq = torch.cat(outputs, dim=1)  # (B,168,s)
        return y_seq


class AttrSeq2SeqBiLSTM(nn.Module):
    """AttrSeq2SeqLSTM_v1 with a BIDIRECTIONAL LSTM encoder. The decoder stays unidirectional
    (autoregressive over the future). The encoder's forward+backward final states are bridged
    (concat -> Linear) down to the decoder's hidden size to initialize it. Everything else
    (per-day forecast injection, flag, lead-time, last_val) matches AttrSeq2SeqLSTM_v1."""
    def __init__(self, enc_in_dim, dec_tf_dim, attr_size,
                 hidden_dim=128, layers=2, out_len=168, dropout=0.2,
                 station_dim=3, forecast_dim=None, forecast_days=4):
        super().__init__()
        self.model_type = 'seq2seq_bilstm'
        self.out_len = out_len
        self.attr_size = attr_size
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.forecast_dim = forecast_dim               # per-day forecast feature dim
        self.forecast_days = forecast_days
        _fct_step_dim = (forecast_dim + 2) if forecast_dim is not None else 0

        print(f"ENCODER DIM: {enc_in_dim}; DECODER DIM: {attr_size + dec_tf_dim + _fct_step_dim}")

        # bidirectional encoder: per-direction hidden is hidden_dim, so encoder outputs 2*hidden_dim
        self.encoder = nn.LSTM(enc_in_dim, hidden_dim, layers, batch_first=True, dropout=dropout, bidirectional=True)
        # bridge fwd+bwd final states (2*hidden) -> hidden, to init the unidirectional decoder
        self.h_bridge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.c_bridge = nn.Linear(2 * hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(attr_size + dec_tf_dim + _fct_step_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, attr_size),
        )
        # full multi-day forecast (flattened) -> initial decoder state for global multi-day context
        self.forecast_proj = nn.Linear(forecast_dim * forecast_days, hidden_dim) if self.forecast_dim is not None else None

    def _bridge(self, state, bridge):
        # state: (layers*2, B, H); even rows = forward layers, odd rows = backward layers
        fwd = state[0::2]                             # (layers, B, H)
        bwd = state[1::2]                             # (layers, B, H)
        return bridge(torch.cat([fwd, bwd], dim=-1))  # (layers, B, H)

    def forward(self, enc_seq, dec_time_feats, last_val=None, station_feats=None, forecast=None):
        B = enc_seq.size(0)

        # Bidirectional encode, then bridge fwd+bwd final states to init the unidirectional decoder
        _, (h, c) = self.encoder(enc_seq)             # h, c: (layers*2, B, H)
        h = self._bridge(h, self.h_bridge)            # (layers, B, H)
        c = self._bridge(c, self.c_bridge)            # (layers, B, H)

        if self.forecast_dim is not None and forecast is not None:
            # forecast: (B, n_days, day_dim); day d is the forecast for (current_day + 1 + d) (excludes current day)
            n_days = forecast.size(1)
            day_dim = forecast.size(2)
            f = self.forecast_proj(forecast.reshape(B, n_days * day_dim))
            h = h + f.unsqueeze(0)
            c = c + f.unsqueeze(0)
            # per-step forecast day aligned via midnight detection (minute-of-day cols 0,1 == (0,1))
            is_midnight = (dec_time_feats[:, :, 0] == 0) & (dec_time_feats[:, :, 1] == 1)   # (B, out_len)
            day_ahead = is_midnight.to(forecast.dtype).cumsum(dim=1)
            fday = day_ahead - 1.0                                                          # day_ahead 1 -> forecast day 0
            valid = (fday >= 0) & (fday <= (n_days - 1))
            gather_idx = fday.clamp(0, n_days - 1).long().unsqueeze(-1).expand(B, self.out_len, day_dim)
            per_step = torch.gather(forecast, 1, gather_idx)                                # (B, out_len, day_dim)
            flag = valid.to(forecast.dtype).unsqueeze(-1)                                   # (B, out_len, 1)
            per_step = per_step * flag
            lead = (day_ahead / (self.out_len / 24.0)).unsqueeze(-1)                        # (B, out_len, 1)
            per_step = torch.cat([per_step, flag, lead], dim=-1)                            # (B, out_len, day_dim+2)
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :], per_step[:, t:t+1, :]]
        else:
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :]]

        last_known_values = last_val if last_val is not None else enc_seq[:, -1:, -self.attr_size:]
        outputs = []
        dec_h, dec_c = h, c
        dec_in = torch.cat(dec_parts(last_known_values, 0), dim=-1)

        for t in range(self.out_len):
            out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))   # out: (B,1,H)
            y_hat = self.proj(out)
            outputs.append(y_hat)
            prev = y_hat
            if t + 1 < self.out_len:
                dec_in = torch.cat(dec_parts(prev, t + 1), dim=-1)

        return torch.cat(outputs, dim=1)  # (B, out_len, attr_size)


class AttrSeq2SeqAttnLSTM(nn.Module):
    """AttrSeq2SeqLSTM_v1 + attention over the encoder outputs (no CNN).
    Identical to v1 (per-day forecast injection, has-forecast flag, lead-time, last_val),
    except the encoder keeps its full output sequence and, at each decoder step, the decoder
    output attends over the (downsampled) encoder memory via scaled dot-product attention.
    The context is fused into the decoder output residually (out + ctx) before projection."""
    def __init__(self, enc_in_dim, dec_tf_dim, attr_size,
                 hidden_dim=128, layers=2, out_len=168, dropout=0.2,
                 station_dim=3, forecast_dim=None, forecast_days=4, attn_stride=2):
        super().__init__()
        self.model_type = 'seq2seq_attn'
        self.out_len = out_len
        self.attr_size = attr_size
        self.attn_stride = attn_stride                 # downsample encoder memory by this factor for attention
        self.attn_scale = hidden_dim ** -0.5           # scaled dot-product

        self.forecast_dim = forecast_dim               # per-day forecast feature dim
        self.forecast_days = forecast_days
        # per decoder step we append the relevant day's forecast slice + has-forecast flag + lead-time (+2)
        _fct_step_dim = (forecast_dim + 2) if forecast_dim is not None else 0

        print(f"ENCODER DIM: {enc_in_dim}; DECODER DIM: {attr_size + dec_tf_dim + _fct_step_dim}")

        self.encoder = nn.LSTM(enc_in_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(attr_size + dec_tf_dim + _fct_step_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, attr_size),
        )
        # full multi-day forecast (flattened) -> initial decoder state for global multi-day context
        self.forecast_proj = nn.Linear(forecast_dim * forecast_days, hidden_dim) if self.forecast_dim is not None else None

    def forward(self, enc_seq, dec_time_feats, last_val=None, station_feats=None, forecast=None):
        B = enc_seq.size(0)

        # Encode: keep the full output sequence for attention
        enc_out, (h, c) = self.encoder(enc_seq)                 # enc_out: (B, T, H)
        enc_mem = enc_out[:, ::self.attn_stride, :]             # (B, T', H) downsampled keys/values
        enc_mem_t = enc_mem.transpose(1, 2)                     # (B, H, T') precomputed for scores

        if self.forecast_dim is not None and forecast is not None:
            # forecast: (B, n_days, day_dim); day d is the forecast for (current_day + 1 + d) (excludes current day)
            n_days = forecast.size(1)
            day_dim = forecast.size(2)
            f = self.forecast_proj(forecast.reshape(B, n_days * day_dim))
            h = h + f.unsqueeze(0)
            c = c + f.unsqueeze(0)
            # per-step forecast day aligned via midnight detection (minute-of-day cols 0,1 == (0,1))
            is_midnight = (dec_time_feats[:, :, 0] == 0) & (dec_time_feats[:, :, 1] == 1)   # (B, out_len)
            day_ahead = is_midnight.to(forecast.dtype).cumsum(dim=1)
            fday = day_ahead - 1.0                                                          # day_ahead 1 -> forecast day 0
            valid = (fday >= 0) & (fday <= (n_days - 1))
            gather_idx = fday.clamp(0, n_days - 1).long().unsqueeze(-1).expand(B, self.out_len, day_dim)
            per_step = torch.gather(forecast, 1, gather_idx)                                # (B, out_len, day_dim)
            flag = valid.to(forecast.dtype).unsqueeze(-1)                                   # (B, out_len, 1)
            per_step = per_step * flag
            lead = (day_ahead / (self.out_len / 24.0)).unsqueeze(-1)                        # (B, out_len, 1)
            per_step = torch.cat([per_step, flag, lead], dim=-1)                            # (B, out_len, day_dim+2)
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :], per_step[:, t:t+1, :]]
        else:
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :]]

        last_known_values = last_val if last_val is not None else enc_seq[:, -1:, -self.attr_size:]
        outputs = []
        dec_h, dec_c = h, c
        dec_in = torch.cat(dec_parts(last_known_values, 0), dim=-1)

        for t in range(self.out_len):
            out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))   # out: (B,1,H)
            # scaled dot-product attention: query=out, keys/values=enc_mem
            scores = torch.bmm(out, enc_mem_t) * self.attn_scale         # (B,1,T')
            weights = torch.softmax(scores, dim=-1)                      # (B,1,T')
            ctx = torch.bmm(weights, enc_mem)                           # (B,1,H)
            y_hat = self.proj(out + ctx)                               # residual fuse of decoder state + context
            outputs.append(y_hat)
            prev = y_hat
            if t + 1 < self.out_len:
                dec_in = torch.cat(dec_parts(prev, t + 1), dim=-1)

        return torch.cat(outputs, dim=1)  # (B, out_len, attr_size)



class AttrSeq2SeqCNNLSTM(nn.Module):
    """Stacked Conv1d + MaxPool1d encoder front-end followed by an autoregressive LSTM decoder.
    Each entry in cnn_channels adds one Conv→ReLU→MaxPool block, progressively compressing the
    sequence before the LSTM. Station features and forecast injected into encoder h/c and
    appended to each decoder step, mirroring AttrSeq2SeqLSTM_v1."""
    def __init__(self, enc_in_dim, dec_tf_dim, attr_size,
                 hidden_dim=128, layers=2, out_len=168, dropout=0.2,
                 station_dim=3, forecast_dim=None, forecast_days=4,
                 cnn_channels=(64, 128, 256), cnn_kernel=3, pool_size=2, dilation=1, batchnorm=False):
        super().__init__()
        self.model_type = 'seq2seq_cnn'
        self.layers = layers
        self.hidden_dim = hidden_dim

        self.forecast_dim = forecast_dim          # per-day forecast feature dim
        self.forecast_days = forecast_days
        # per decoder step we append the relevant day's forecast slice + has-forecast flag + lead-time (+2)
        _fct_step_dim = (forecast_dim + 2) if forecast_dim is not None else 0
        print(f"ENCODER DIM: {enc_in_dim}; DECODER DIM: {attr_size + dec_tf_dim + _fct_step_dim}")

        self.out_len = out_len
        self.attr_size = attr_size
        blocks = []
        in_ch = enc_in_dim
        for i, out_ch in enumerate(cnn_channels):
            d = dilation ** i                       # per-layer dilation: 1, dilation, dilation^2, ...
            pad = d * (cnn_kernel - 1) // 2         # 'same' padding (keeps length) for this dilation
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel, padding=pad, dilation=d),
                nn.ReLU(),
            ]
            if pool_size and pool_size > 1:
                blocks += [nn.MaxPool1d(pool_size)]  # set pool_size=1 to disable downsampling
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        self.encoder = nn.LSTM(in_ch, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(attr_size + dec_tf_dim + _fct_step_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        # self.decoder = nn.LSTM(attr_size + dec_tf_dim + _fct_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.forecast_proj = nn.Linear(forecast_dim * forecast_days, hidden_dim) if self.forecast_dim is not None else None
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, attr_size),
        )

    def forward(self, enc_seq, dec_time_feats, last_val=None, station_feats=None, forecast=None):
        B = enc_seq.size(0)
        # CNN: (B, T, C) -> (B, C, T) -> conv/pool -> (B, T', cnn_filters)
        x = self.cnn(enc_seq.float().permute(0, 2, 1)).permute(0, 2, 1)
        _, (h, c) = self.encoder(x)

        if self.forecast_dim is not None and forecast is not None:
            # forecast: (B, n_days, day_dim); day d (0-indexed) is the forecast for (current_day + 1 + d),
            # i.e. the forecast does NOT cover the current day.
            n_days = forecast.size(1)
            day_dim = forecast.size(2)
            # global context: full multi-day forecast (flattened) into the initial decoder state
            f = self.forecast_proj(forecast.reshape(B, n_days * day_dim))
            h = h + f.unsqueeze(0)
            c = c + f.unsqueeze(0)

            # Align each horizon step to the right forecast day. A new calendar day is the midnight step,
            # detected directly from minute-of-day (cols 0,1 = sin,cos): at 00:00 sin==0 and cos==1.
            # Counting midnights from step 0 gives day_ahead (no input-end anchor needed);
            # day_ahead==1 -> forecast day 0 (forecast excludes current day). This also handles the
            # year/leap boundary that a day-of-year change check would miss.
            is_midnight = (dec_time_feats[:, :, 0] == 0) & (dec_time_feats[:, :, 1] == 1)   # (B, out_len)
            day_ahead = is_midnight.to(forecast.dtype).cumsum(dim=1)                        # midnights crossed
            fday = day_ahead - 1.0                                            # day_ahead 1 -> forecast day 0
            valid = (fday >= 0) & (fday <= (n_days - 1))                                               # (B, out_len)
            gather_idx = fday.clamp(0, n_days - 1).long().unsqueeze(-1).expand(B, self.out_len, day_dim)
            per_step = torch.gather(forecast, 1, gather_idx)                                           # (B, out_len, day_dim)
            flag = valid.to(forecast.dtype).unsqueeze(-1)                                              # (B, out_len, 1)
            per_step = per_step * flag                                                                 # zero out non-covered days
            # normalized forecast lead time (days ahead of the last observation): lets the model tell
            # near-term (rest of today, day_ahead 0) from far future (days 5-7), which both have flag=0.
            lead = (day_ahead / (self.out_len / 24.0)).unsqueeze(-1)                                   # (B, out_len, 1)
            per_step = torch.cat([per_step, flag, lead], dim=-1)                                       # (B, out_len, day_dim+2)
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :], per_step[:, t:t+1, :]]
            # print('prev_doy', prev_doy[0])
            # print('day_changed', day_changed[0])
            # print('day_ahead', day_ahead[0])
            # print('per_step', per_step[0])
            # print('flag', flag[0])
        else:
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :]]
        last_known_values = last_val if last_val is not None else enc_seq[:, -1:, -self.attr_size:]
        outputs = []
        dec_h, dec_c = h, c
        dec_in = torch.cat(dec_parts(last_known_values, 0), dim=-1)
        for t in range(self.out_len):
            out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))
            y_hat = self.proj(out)
            outputs.append(y_hat)
            if t + 1 < self.out_len:
                dec_in = torch.cat(dec_parts(y_hat, t + 1), dim=-1)
        return torch.cat(outputs, dim=1)  # (B, out_len, attr_size)


class AttrSeq2SeqCNNAttnLSTM(nn.Module):
    """AttrSeq2SeqCNNLSTM + AttrSeq2SeqAttnLSTM: Conv1d/MaxPool encoder front-end AND attention
    over the encoder outputs. The CNN blocks compress the input sequence, the encoder LSTM keeps
    its full output sequence over that compressed representation, and at each decoder step the
    decoder output attends over the (optionally further downsampled) encoder memory via scaled
    dot-product attention. The context is fused residually (out + ctx) before projection.
    Forecast/lead-time handling is identical to both parents (AttrSeq2SeqLSTM_v1 semantics).
    Note: the CNN already shortens the sequence by pool_size**len(cnn_channels), so attn_stride
    defaults to 1 (no extra downsampling of the attention memory)."""
    def __init__(self, enc_in_dim, dec_tf_dim, attr_size,
                 hidden_dim=128, layers=2, out_len=168, dropout=0.2,
                 station_dim=3, forecast_dim=None, forecast_days=4,
                 cnn_channels=(64, 128, 256), cnn_kernel=3, pool_size=2, dilation=1, batchnorm=False,
                 attn_stride=1):
        super().__init__()
        self.model_type = 'seq2seq_cnn_attn'
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.out_len = out_len
        self.attr_size = attr_size
        self.attn_stride = attn_stride                 # extra downsampling of encoder memory for attention
        self.attn_scale = hidden_dim ** -0.5           # scaled dot-product

        self.forecast_dim = forecast_dim               # per-day forecast feature dim
        self.forecast_days = forecast_days
        # per decoder step we append the relevant day's forecast slice + has-forecast flag + lead-time (+2)
        _fct_step_dim = (forecast_dim + 2) if forecast_dim is not None else 0
        print(f"ENCODER DIM: {enc_in_dim}; DECODER DIM: {attr_size + dec_tf_dim + _fct_step_dim}")

        blocks = []
        in_ch = enc_in_dim
        for i, out_ch in enumerate(cnn_channels):
            d = dilation ** i                       # per-layer dilation: 1, dilation, dilation^2, ...
            pad = d * (cnn_kernel - 1) // 2         # 'same' padding (keeps length) for this dilation
            blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel, padding=pad, dilation=d),
                nn.ReLU(),
            ]
            if pool_size and pool_size > 1:
                blocks += [nn.MaxPool1d(pool_size)]  # set pool_size=1 to disable downsampling
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)
        self.encoder = nn.LSTM(in_ch, hidden_dim, layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(attr_size + dec_tf_dim + _fct_step_dim, hidden_dim, layers, batch_first=True, dropout=dropout)
        # full multi-day forecast (flattened) -> initial decoder state for global multi-day context
        self.forecast_proj = nn.Linear(forecast_dim * forecast_days, hidden_dim) if self.forecast_dim is not None else None
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, attr_size),
        )

    def forward(self, enc_seq, dec_time_feats, last_val=None, station_feats=None, forecast=None):
        B = enc_seq.size(0)

        # CNN: (B, T, C) -> (B, C, T) -> conv/pool -> (B, T', cnn_filters)
        x = self.cnn(enc_seq.float().permute(0, 2, 1)).permute(0, 2, 1)
        # Encode: keep the full output sequence for attention
        enc_out, (h, c) = self.encoder(x)                       # enc_out: (B, T', H)
        enc_mem = enc_out[:, ::self.attn_stride, :]             # (B, T'', H) keys/values
        enc_mem_t = enc_mem.transpose(1, 2)                     # (B, H, T'') precomputed for scores

        if self.forecast_dim is not None and forecast is not None:
            # forecast: (B, n_days, day_dim); day d is the forecast for (current_day + 1 + d) (excludes current day)
            n_days = forecast.size(1)
            day_dim = forecast.size(2)
            f = self.forecast_proj(forecast.reshape(B, n_days * day_dim))
            h = h + f.unsqueeze(0)
            c = c + f.unsqueeze(0)
            # per-step forecast day aligned via midnight detection (minute-of-day cols 0,1 == (0,1))
            is_midnight = (dec_time_feats[:, :, 0] == 0) & (dec_time_feats[:, :, 1] == 1)   # (B, out_len)
            day_ahead = is_midnight.to(forecast.dtype).cumsum(dim=1)
            fday = day_ahead - 1.0                                                          # day_ahead 1 -> forecast day 0
            valid = (fday >= 0) & (fday <= (n_days - 1))
            gather_idx = fday.clamp(0, n_days - 1).long().unsqueeze(-1).expand(B, self.out_len, day_dim)
            per_step = torch.gather(forecast, 1, gather_idx)                                # (B, out_len, day_dim)
            flag = valid.to(forecast.dtype).unsqueeze(-1)                                   # (B, out_len, 1)
            per_step = per_step * flag                                                      # zero out non-covered days
            lead = (day_ahead / (self.out_len / 24.0)).unsqueeze(-1)                        # (B, out_len, 1)
            per_step = torch.cat([per_step, flag, lead], dim=-1)                            # (B, out_len, day_dim+2)
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :], per_step[:, t:t+1, :]]
        else:
            dec_parts = lambda prev, t: [prev, dec_time_feats[:, t:t+1, :]]

        last_known_values = last_val if last_val is not None else enc_seq[:, -1:, -self.attr_size:]
        outputs = []
        dec_h, dec_c = h, c
        dec_in = torch.cat(dec_parts(last_known_values, 0), dim=-1)

        for t in range(self.out_len):
            out, (dec_h, dec_c) = self.decoder(dec_in, (dec_h, dec_c))   # out: (B,1,H)
            # scaled dot-product attention: query=out, keys/values=enc_mem
            scores = torch.bmm(out, enc_mem_t) * self.attn_scale         # (B,1,T'')
            weights = torch.softmax(scores, dim=-1)                      # (B,1,T'')
            ctx = torch.bmm(weights, enc_mem)                            # (B,1,H)
            y_hat = self.proj(out + ctx)                                 # residual fuse of decoder state + context
            outputs.append(y_hat)
            if t + 1 < self.out_len:
                dec_in = torch.cat(dec_parts(y_hat, t + 1), dim=-1)

        return torch.cat(outputs, dim=1)  # (B, out_len, attr_size)




###############################################
############ WITH station_id_match ############
###############################################

