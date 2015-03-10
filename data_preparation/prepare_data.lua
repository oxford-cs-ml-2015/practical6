require 'torch'
local CharLMMinibatchLoader=require 'data.CharLMMinibatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert data to torch format')
cmd:text()
cmd:text('Options')
cmd:option('-txt','input.txt','data source')
cmd:option('-vocab','vocab.t7','name of the char->int table to save')
cmd:option('-data','data.t7','name of the serialized torch ByteTensor to save')
cmd:text()

params = cmd:parse(arg)
CharLMMinibatchLoader.text_to_tensor(params.txt, params.vocab, params.data)

