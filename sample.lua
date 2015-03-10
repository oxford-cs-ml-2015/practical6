require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'Embedding'                     -- class name is Embedding (not namespaced)


cmd = torch.CmdLine()
cmd:text()
cmd:text('Test a simple character-level LSTM language model')
cmd:text()
cmd:text('Options')
cmd:option('-vocabfile','vocabfile.t7','filename of the string->int table')
cmd:option('-model','model_file.t7','contains just the protos table, and nothing else')
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',false,'false to use max at each timestep, true to sample at each timestep')
cmd:option('-primetext',"hello my name is ",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample. set to a space " " to disable')
cmd:option('-length',200,'number of characters to sample')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- preparation and loading
torch.manualSeed(opt.seed)

local vocab = torch.load(opt.vocabfile)
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- load model and recreate a few important numbers
protos = torch.load(opt.model)
opt.rnn_size = protos.embed.weight:size(2)

--protos.embed = Embedding(vocab_size, opt.rnn_size)
---- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
--protos.lstm = LSTM.lstm(opt)
--protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, vocab_size)):add(nn.LogSoftMax())
--protos.criterion = nn.ClassNLLCriterion()

-- LSTM initial state, note that we're using minibatches OF SIZE ONE here
local prev_c = torch.zeros(1, opt.rnn_size)
local prev_h = prev_c:clone()

local seed_text = opt.primetext
local prev_char

-- do some seeded timesteps
for c in seed_text:gmatch'.' do
    prev_char = torch.Tensor{vocab[c]}

    local embedding = protos.embed:forward(prev_char)
    local next_c, next_h = unpack(protos.lstm:forward{embedding, prev_c, prev_h})

    prev_c:copy(next_c) -- TODO: this shouldn't be needed... check if we can just use an assignment?
    prev_h:copy(next_h)
end

-- now start sampling/argmaxing
for i=1, opt.length do
    -- embedding and LSTM 
    local embedding = protos.embed:forward(prev_char)
    local next_c, next_h = unpack(protos.lstm:forward{embedding, prev_c, prev_h})
    prev_c:copy(next_c)
    prev_h:copy(next_h)
    
    -- softmax from previous timestep
    local log_probs = protos.softmax:forward(next_h)

    if not opt.sample then
        -- use argmax
        local _, prev_char_ = log_probs:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        local probs = torch.exp(log_probs):squeeze()
        prev_char = torch.multinomial(probs, 1):resize(1)
    end

    --print('OUTPUT:', ivocab[prev_char[1]])
    io.write(ivocab[prev_char[1]])
end
io.write('\n') io.flush()

