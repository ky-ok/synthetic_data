本レポジトリは、https://note.com/hatti8/n/n193430331561 にて、合成データ生成方法の基本形として、
紹介いただいた「Self-Instruct」という手法（https://arxiv.org/abs/2212.10560）を
https://github.com/Hajime-Y/Alpaca-jp のコードを参考に、実装したものです。(はちさん、ありがとうございます。)

参照先のレポジトリ(https://github.com/Hajime-Y/Alpaca-jp)では、Mistral-8x22B, Mistral-8x7Bを用いていましたが、
本レポジトリでは、Qwen/Qwen2.5-32B-Instructを用いて、合成データを生成できるよう変更しました。

## Usage

- 実行サンプルファイル(実行環境：Google Colaboratory)：[1_generate_instruction_sample.ipynb](https://github.com/ky-ok/synthetic_data/blob/main/1_generate_instruction_sample.ipynb)

 ※ カレントディレクトリに、「seed_tasks」という名前のディレクトリを作成し、ディレクトリ内にSeed tasksファイル（本環境のファイル名は、「Elyza-tasks-100_seed_tasks.jsonl」）を格納して、実行しています。

## Prompt

データ合成のためのプロンプト([prompt_en_for_jp.txt](https://github.com/ky-ok/synthetic_data/blob/develop/prompt_en_for_jp.txt))は、
参照先のレポジトリ(https://github.com/Hajime-Y/Alpaca-jp)から、持ってきました。
Qwen2.5も、日本語の出力は、英語のinstructionの方が効果的とあったので、英語のプロンプトを使用しました。