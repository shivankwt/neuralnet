require("math")
math.randomseed(os.time())

local inputs = 2
local hiddenNodes = 2
local outputs = 1
local training = 4

local function initWeight()
	return (math.random() * 2) - 1 -- b/w 1 -1
end

local function sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

local function derivative(x)
	return x * (1 - x)
end

local function shuffle(table)
	for i = #table, 2, -1 do 
		local j = math.random(1, i)
		table[i], table[j] = table[j], table[i]
	end
end

local function initModel()
	local model = {
		rate = 0.1,
		hiddenLayer = {},
		outputLayer = {},

		hiddenLayerBias = {},
		outputLayerBias = {},

		hiddenWeights = {},
		outputWeights = {},

		trainingSetOrder = { 1, 2, 3, 4 },
		numberOfEpochs = 10000,

		trainingInputs = {
			{ 0.0, 0.0 },
			{ 1.0, 0.0 },
			{ 0.0, 1.0 },
			{ 1.0, 1.0 }
		},

		trainingOutputs = {
			{ 0.0 },
			{ 1.0 },
			{ 1.0 },
			{ 0.0 }
		}
	}

	for i = 1, hiddenNodes do
		model.hiddenLayer[i] = 0
		model.hiddenLayerBias[i] = initWeight()
	end
	
	for i = 1, outputs do
		model.outputLayer[i] = 0
		model.outputLayerBias[i] = initWeight()
	end

	for i = 1, inputs do
		model.hiddenWeights[i] = {}
		for j = 1, hiddenNodes do 
			model.hiddenWeights[i][j] = initWeight()
		end
	end

	for i = 1, hiddenNodes do 
		model.outputWeights[i] = {}
		for j = 1, outputs do 
			model.outputWeights[i][j] = initWeight()
		end
	end

	return model
end

local function displayResult()
	local model = initModel()

	for epoch = 1, model.numberOfEpochs do 
		if epoch % 1000 == 0 then
			print(epoch, "simulations done")
		end

		shuffle(model.trainingSetOrder)
	
		for x = 1, #model.trainingSetOrder do
			local i = model.trainingSetOrder[x]

			-- forward pass: compute hidden layer
			for j = 1, hiddenNodes do 
				local activation = model.hiddenLayerBias[j]
				for k = 1, inputs do 
					activation = activation + model.trainingInputs[i][k] * model.hiddenWeights[k][j]
				end
				model.hiddenLayer[j] = sigmoid(activation)
			end

			-- forward pass: compute output layer
			for j = 1, outputs do 
				local activation = model.outputLayerBias[j]
				for k = 1, hiddenNodes do 
					activation = activation + model.hiddenLayer[k] * model.outputWeights[k][j]
				end
				model.outputLayer[j] = sigmoid(activation)
			end

			-- backward pass: compute delta for output layer
			local deltaOutput = {}
			for j = 1, outputs do 
				local err = model.trainingOutputs[i][j] - model.outputLayer[j]
				deltaOutput[j] = err * derivative(model.outputLayer[j])
			end

			-- backward pass: compute delta for hidden layer
			local deltaHidden = {}
			for j = 1, hiddenNodes do 
				local err = 0.0
				for k = 1, outputs do 
					err = err + deltaOutput[k] * model.outputWeights[j][k]
				end
				deltaHidden[j] = err * derivative(model.hiddenLayer[j])
			end

			-- update output weights and biases
			for j = 1, outputs do 
				model.outputLayerBias[j] = model.outputLayerBias[j] + deltaOutput[j] * model.rate
				for k = 1, hiddenNodes do 
					model.outputWeights[k][j] = model.outputWeights[k][j] + model.hiddenLayer[k] * deltaOutput[j] * model.rate
				end
			end
			
			-- update hidden weights and biases
			for j = 1, hiddenNodes do 
				model.hiddenLayerBias[j] = model.hiddenLayerBias[j] + deltaHidden[j] * model.rate
				for k = 1, inputs do 
					model.hiddenWeights[k][j] = model.hiddenWeights[k][j] + model.trainingInputs[i][k] * deltaHidden[j] * model.rate
				end
			end
		end
	end

	for i = 1, training do 		
		for j = 1, hiddenNodes do
			local activation = model.hiddenLayerBias[j]
			for k = 1, inputs do
				activation = activation + model.trainingInputs[i][k] * model.hiddenWeights[k][j]
			end
			model.hiddenLayer[j] = sigmoid(activation)
		end

		for j = 1, outputs do
			local activation = model.outputLayerBias[j]
			for k = 1, hiddenNodes do
				activation = activation + model.hiddenLayer[k] * model.outputWeights[k][j]
			end
			model.outputLayer[j] = sigmoid(activation)
		end

		print(string.format(
			"input values: %f %f || target: %f || prediction: %f",
			model.trainingInputs[i][1],
			model.trainingInputs[i][2],
			model.trainingOutputs[i][1],
			model.outputLayer[1]
		))
	end
end

displayResult()

