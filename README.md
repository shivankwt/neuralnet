#### neuralnet

it's a simple neural network in lua. it learns XOR operation with two inputs, two hidden nodes and one output

* how does it work?

if you don't know neural network then watch this [video](https://youtu.be/aircAruvnKk?si=vlv2XDY9oIBd4zlH),
although what i created is slightly different, nevertheless given video builds better intuition regarding what
a neural network is. i used *lua* programming language because i wanted to try something new.

>p.s. this is written in **pure lua**. [ no external dependencies ].

this neural network is built on *two inputs*, *two hidden nodes* and provides *one output*.

* what does it do?

This neural network uses XOR training data to learn the relationship between the inputs and output, adjusting its
weights and biases through backpropagation until it can accurately predict the xor result for all possible input pairs.

* XOR Training Data

| Input A | Input B | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 1       | 0       | 1      |
| 0       | 1       | 1      |
| 1       | 1       | 0      |

