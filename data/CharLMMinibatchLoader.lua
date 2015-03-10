-- loader for character-level language models

require 'torch'
require 'math'

local CharLMMinibatchLoader = {}
CharLMMinibatchLoader.__index = CharLMMinibatchLoader

function CharLMMinibatchLoader.create(tensor_file, vocab_file, batch_size, seq_length)
    local self = {}
    setmetatable(self, CharLMMinibatchLoader)

    -- construct a tensor with all the data
    print('loading data files...')
    local data = torch.load(tensor_file)
    self.vocab_mapping = torch.load(vocab_file)

    -- cut off the end so that it divides evenly
    local len = data:size(1)
    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        data = data:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end

    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end

    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = data:clone()
    ydata:sub(1,-2):copy(data:sub(2,-1))
    ydata[-1] = data[1]
    self.x_batches = data:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    assert(#self.x_batches == #self.y_batches)

    self.current_batch = 0
    self.evaluated_batches = 0  -- number of times next_batch() called

    print('data load done.')
    collectgarbage()
    return self
end

-- *** STATIC method ***
function CharLMMinibatchLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile)
    local timer = torch.Timer()

    print('timer: ', timer:time().real)
    print('loading text file...')
    local f = torch.DiskFile(in_textfile)
    local rawdata = f:readString('*a') -- NOTE: this reads the whole file at once
    f:close()

    -- create vocabulary if it doesn't exist yet
    print('timer: ', timer:time().real)
    print('creating vocabulary mapping...')
    -- record all of them into a set
    local unordered = {}
    for char in rawdata:gmatch'.' do
        if not unordered[char] then unordered[char] = true end
    end

    -- sort them
    local ordered = {}
    for char in pairs(unordered) do ordered[#ordered + 1] = char end
    table.sort(ordered) -- now order maps int->char

    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i, char in ipairs(ordered) do
        vocab_mapping[char] = i
    end

    -- construct a tensor with all the data
    print('timer: ', timer:time().real)
    print('putting data into tensor...')
    local data = torch.ByteTensor(#rawdata) -- store it into 1D first, then rearrange
    for i=1, #rawdata do
        data[i] = vocab_mapping[rawdata:sub(i, i)] -- lua has no string indexing using []
    end

    print('saving two files...')
    torch.save(out_vocabfile, vocab_mapping)
    torch.save(out_tensorfile, data)

    print('Done in time (seconds): ', timer:time().real)
end

function CharLMMinibatchLoader:next_batch()
    self.current_batch = (self.current_batch % self.nbatches) + 1
    self.evaluated_batches = self.evaluated_batches + 1
    return self.x_batches[self.current_batch], self.y_batches[self.current_batch]
end

return CharLMMinibatchLoader

