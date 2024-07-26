import os

import torch
import wandb


def train_epoch(
    train_loader,
    model,
    optimizer,
    loss_fn,
):
    running_loss = 0.0
    for idx, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1)

        loss = loss_fn(outputs, labels)
        loss_value = loss.item()
        loss.backward()

        running_loss += loss_value

        optimizer.step()

    return running_loss / len(train_loader)


def validate_epoch(
    val_loader,
    model,
    loss_fn,
):
    running_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data

            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            labels = labels.squeeze(1)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

        return running_loss / len(val_loader)


def train(
    epochs,
    train_loader,
    val_loader,
    model,
    optimizer,
    loss_fn,
    save_path,
    use_wandb=True,
    maximize=False,
):
    epoch_number = 0
    best_val_loss = None

    for epoch in range(epochs):
        print(f"\n\n\n\nEPOCH {epoch}\n\n")

        model.train()
        avg_train_loss = train_epoch(train_loader, model, optimizer, loss_fn)

        print(f"\n\ntrain loss:\t{avg_train_loss}")

        model.eval()
        avg_val_loss = validate_epoch(val_loader, model, loss_fn)

        print(f"val loss:\t{avg_val_loss}")

        epoch_number += 1

        if use_wandb:
            wandb.log(
                {
                    "average training loss": avg_train_loss,
                    "average validation loss": avg_val_loss,
                }
            )

        if epoch_number % 5 == 0 or epoch_number == 1 or epoch_number == epochs:
            path = os.path.join(save_path, f"epoch_{epoch_number}")
            torch.save(model.state_dict(), path)

        if (
            best_val_loss is None
            or (maximize and avg_val_loss > best_val_loss)
            or (not maximize and avg_val_loss < best_val_loss)
        ):
            best_val_loss = avg_val_loss
            path = os.path.join(save_path, "best")
            print(f"saving best on epoch {epoch_number}")
            torch.save(model.state_dict(), path)
