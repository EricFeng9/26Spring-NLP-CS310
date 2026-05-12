
- 环境变量 `MODEL=/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Lab/week12/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf`

- 测试命令 `./llama.cpp/build/bin/llama-cli -m "$MODEL" -p "2+2= ?"

# GGUF 模型（给 llama.cpp 用）
export MODEL_GGUF=/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Lab/week12/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf

# transformers 原始模型目录（给 notebook 用）
export MODEL_QWEN3=/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Lab/week12/models/Qwen3-1.7B
export MODEL_QWEN25=/data/student/Fengjunming/Temp/26Spring-NLP-CS310/Lab/week12/models/Qwen2.5-7B
`

# T1

## 1.1 zeroshot

### 1.1.1
`./llama.cpp/build/bin/llama-cli -m "$MODEL_GGUF" -p "请只输出最终答案，不要解释。
题目：A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?
答案：
\\nothink"`

- 回答: 4

### 1.1.2
`./llama.cpp/build/bin/llama-cli -m "$MODEL_GGUF" -p "请只输出最终答案，不要解释。
题目：鸡和兔在一个笼子里，共有35个头，94只脚，那么鸡有多少只，兔有多少只？
答案：
\nothink"`

- 回答:17鸡，18兔

### 1.1.3
`./llama.cpp/build/bin/llama-cli -m "$MODEL_GGUF" -p "请只输出最终答案，不要解释。
题目：242342 + 423443 = ?
答案：
\nothink"`

- 回答:2423423443

### 1.1.4
`./llama.cpp/build/bin/llama-cli -m "$MODEL_GGUF" -p "请只输出最终答案，不要解释。
题目：一个人花8块钱买了一只鸡，9块钱卖掉了，然后他觉得不划算，花10块钱又买回来了，11块卖给另外一个人。问他赚了多少?
答案：
\nothink"`

- 回答: 3块钱

## 1.2  few-shot 

### 1.2.1
`请根据示例作答，只输出最终答案，不要解释。
示例1：
题目：一个数加上3等于8，求这个数。
答案：5

示例2：
题目：20个球里一半是红球，红球里一半是大球，大红球有几个？
答案：5

现在回答：
题目：Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: 
答案：
\nothink`

- 回答: 4

### 1.2.2
`请根据示例作答，只输出最终答案，不要解释。
示例1：
题目：一个数加上3等于8，求这个数。
答案：5

示例2：
题目：20个球里一半是红球，红球里一半是大球，大红球有几个？
答案：5

现在回答：
题目：鸡和兔在一个笼子里，共有35个头，94只脚，那么鸡有多少只，兔有多少只？
答案：
\nothink`

- 回答:

### 1.2.3
`请根据示例作答，只输出最终答案，不要解释。
示例1：
题目：一个数加上3等于8，求这个数。
答案：5

示例2：
题目：20个球里一半是红球，红球里一半是大球，大红球有几个？
答案：5

现在回答：
题目： Q: 242342 + 423443 = ? A: 
答案：
\nothink `

- 回答:2423423443

### 1.2.4
`请根据示例作答，只输出最终答案，不要解释。
示例1：
题目：一个数加上3等于8，求这个数。
答案：5

示例2：
题目：20个球里一半是红球，红球里一半是大球，大红球有几个？
答案：5

现在回答：
题目： 一个人花8块钱买了一只鸡，9块钱卖掉了，然后他觉得不划算，花10块钱又买回来了，11块卖给另外一个人。问他赚了多少?
答案：
\nothink `

- 回答: 3