import torch
import torch.nn as nn
from tqdm import tqdm


def train_model(model, train_loader, epochs=20, lr=1e-4, save_path=None):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()
    model.train()

    print(f"開始訓練程序，共 {epochs} Epochs")
    print("=" * 50)

    loss_history = []

    for epoch in range(epochs):
        total_loss = 0
        last_context = ""

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_seqs, batch_prompts in progress_bar:
            optimizer.zero_grad()

            inputs = batch_seqs.cuda()
            target = inputs[:, -1, :].cuda()
            current_env_text = batch_prompts[0]

            trend = "upward" if inputs.mean() > 0 else "downward"
            _, top_indices = torch.topk(torch.abs(inputs[0, :, 0]), k=5)
            top5_lags = top_indices.tolist()

            pred = model(inputs, trend, top5_lags, expert_knowledge=current_env_text)
            last_context = model.ace.generator(trend, top5_lags, current_env_text)

            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)

        input_x = inputs[0, :, 0].cpu().detach().numpy().tolist()
        pred_out = pred[0].cpu().detach().numpy().tolist()
        actual_out = target[0].cpu().detach().numpy().tolist()

        print(f"\n[Epoch {epoch+1}/{epochs}] Average MSE Loss: {avg_loss:.5f}")
        print("-" * 50)
        print("過去 16 步 X 軸位移（正規化）：")
        print([round(v, 4) for v in input_x])
        print(f"\nACE 動態上下文：\n{last_context}")
        print(f"\n預測：X={pred_out[0]:.4f}  Z={pred_out[1]:.4f}")
        print(f"實際：X={actual_out[0]:.4f}  Z={actual_out[1]:.4f}")
        print("=" * 50)

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"模型已儲存至：{save_path}")

    return loss_history
