import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from data_utils import prepare_training_data
from dataset import LatheDatasetWithPrompt
from model import TimeLLMWithACE
from train import train_model
from inference import generate_answers
from evaluate import calculate_pseudo_rmse


def run_train(args):
    csv_path = prepare_training_data(
        env_excel_path=args.env_excel_path,
        train_folder=args.train_folder,
        save_path=args.train_csv_path,
    )
    dataset = LatheDatasetWithPrompt(csv_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TimeLLMWithACE().cuda()
    train_model(model, loader, epochs=args.epochs, lr=args.lr, save_path=args.model_path)


def run_infer(args):
    train_df = pd.read_csv(args.train_csv_path).dropna()
    train_raw = torch.tensor(train_df[['disp_x_diff', 'disp_z_diff']].values, dtype=torch.float32)
    train_mean = train_raw.mean(dim=0)
    train_std = train_raw.std(dim=0)

    model = TimeLLMWithACE().cuda()
    model.load_state_dict(torch.load(args.model_path))

    generate_answers(
        model=model,
        train_mean=train_mean,
        train_std=train_std,
        env_excel_path=args.env_excel_path,
        test_folder=args.test_folder,
        output_folder=args.output_folder,
    )


def run_evaluate(args):
    calculate_pseudo_rmse(
        original_folder=args.test_folder,
        answer_folder=args.output_folder,
    )


def main():
    parser = argparse.ArgumentParser(description="Time-LLM + ACE 車床熱位移預測")
    parser.add_argument(
        '--mode', required=True, choices=['train', 'infer', 'evaluate'],
        help='執行模式：train（訓練）、infer（推論填答）、evaluate（評估）'
    )
    parser.add_argument('--env_excel_path', type=str, default='/content/drive/MyDrive/Time-LLM-main/檔案環境設定總表.xlsx')
    parser.add_argument('--train_folder',   type=str, default='/content/drive/MyDrive/Time-LLM-main/TRAIN 0-5')
    parser.add_argument('--test_folder',    type=str, default='/content/drive/MyDrive/Time-LLM-main/初賽測驗用數據')
    parser.add_argument('--train_csv_path', type=str, default='/content/drive/MyDrive/train_env.csv')
    parser.add_argument('--model_path',     type=str, default='/content/drive/MyDrive/time_llm_ace_env_v2.pth')
    parser.add_argument('--output_folder',  type=str, default='/content/drive/MyDrive/Time-LLM-main/預測結果輸出')
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--lr',         type=float, default=1e-4)

    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'infer':
        run_infer(args)
    elif args.mode == 'evaluate':
        run_evaluate(args)


if __name__ == '__main__':
    main()
