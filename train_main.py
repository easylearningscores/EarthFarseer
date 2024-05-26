import argparse
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import Earthfarseer_model

def load_navier_stokes_data(path, sub=1, T_in=10, T_out=10, batch_size=2, reshape=None):
    ntrain = 1000
    neval = 100
    ntest = 100
    total = ntrain + neval + ntest
    f = scipy.io.loadmat(path)
    data = f['u'][..., 0:total]
    data = torch.tensor(data, dtype=torch.float32)

    train_a = data[:ntrain, ::sub, ::sub, :T_in]
    train_u = data[:ntrain, ::sub, ::sub, T_in:T_out+T_in]
    train_a = train_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    train_u = train_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)

    eval_a = data[ntrain:ntrain + neval, ::sub, ::sub, :T_in]
    eval_u = data[ntrain:ntrain + neval, ::sub, ::sub, T_in:T_out+T_in]
    eval_a = eval_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    eval_u = eval_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)

    test_a = data[ntrain + neval:ntrain + neval + ntest, ::sub, ::sub, :T_in]
    test_u = data[ntrain + neval:ntrain + neval + ntest, ::sub, ::sub, T_in:T_out+T_in]
    test_a = test_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    test_u = test_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)

    if reshape:
        train_a = train_a.permute(reshape)
        train_u = train_u.permute(reshape)
        eval_a = eval_a.permute(reshape)
        eval_u = eval_u.permute(reshape)
        test_a = test_a.permute(reshape)
        test_u = test_u.permute(reshape)

    train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_a, eval_u), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader

def evaluate_model(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    return total_loss / total_samples

def train_model(model, train_loader, eval_loader, criterion, optimizer, device, num_epochs=150):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            progress_bar.set_postfix(loss=loss.item())

        average_loss = total_loss / total_samples
        print(f'Epoch {epoch + 1}, Train Loss: {average_loss:.7f}')

        eval_loss = evaluate_model(model, eval_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Eval Loss: {eval_loss:.7f}')

        if eval_loss < best_loss:
            best_loss = eval_loss
            print(f'New best model found at epoch {epoch + 1} with loss {best_loss:.7f}. Saving model...')
            torch.save(model.state_dict(), 'best_model_weights.pth')

    print("Training complete.")

def main(args):
    train_loader, eval_loader, test_loader = load_navier_stokes_data(
        path=args.data_path,
        sub=args.sub,
        T_in=args.T_in,
        T_out=args.T_out,
        batch_size=args.batch_size,
        reshape=args.reshape
    )

    model = Earthfarseer_model(shape_in=(10, 1, 64, 64))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_model(model, train_loader, eval_loader, criterion, optimizer, device, num_epochs=args.num_epochs)

    # Load the best model weights for testing
    model.load_state_dict(torch.load('best_model_weights.pth'))

    test_loss = test_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.7f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DGODE model on Navier-Stokes data.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the Navier-Stokes dataset.')
    parser.add_argument('--sub', type=int, default=1, help='Subsampling factor.')
    parser.add_argument('--T_in', type=int, default=10, help='Number of input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Number of output time steps.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--reshape', type=int, nargs='+', help='Optional reshape permutation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs.')

    args = parser.parse_args()
    main(args)
