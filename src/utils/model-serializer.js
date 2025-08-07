/**
 * ModelSerializer - PyTorch-compatible model serialization and loading
 * Handles model state_dict, optimizer states, and training metadata
 */
import logger from './logger.js';
import MemoryManager from './memory-manager.js';

class ModelSerializer {
  constructor(options = {}) {
    this.options = {
      format: options.format || 'json', // 'json' or 'binary'
      compression: options.compression || false,
      includeMetadata: options.includeMetadata !== false,
      validateOnLoad: options.validateOnLoad !== false,
      ...options
    };
    
    this.memoryManager = new MemoryManager();
    this.version = '1.0.0';
  }

  /**
   * Save model state dictionary - PyTorch compatible
   * @param {nn.Module} model - Neural network model
   * @param {string} path - File path or returns serialized data if path is null
   * @param {Object} options - Save options
   * @returns {Promise<Object|string>} Serialized data or file path
   */
  async save(model, path = null, options = {}) {
    try {
      logger.info('Starting model serialization', { path, options });

      // Extract model state dictionary
      const stateDict = await this._extractStateDict(model);
      
      // Create serialized data structure
      const serializedData = {
        version: this.version,
        timestamp: Date.now(),
        model_state_dict: stateDict,
        metadata: this.options.includeMetadata ? this._extractMetadata(model) : null,
        greed_version: '2.0.0', // Current GreedJS version
        pytorch_compatible: true
      };

      // Add custom options
      if (options.epoch) serializedData.epoch = options.epoch;
      if (options.loss) serializedData.loss = options.loss;
      if (options.optimizer_state) {
        serializedData.optimizer_state_dict = await this._extractOptimizerState(options.optimizer_state);
      }

      // Serialize to chosen format
      let serialized;
      if (this.options.format === 'binary') {
        serialized = await this._serializeToBinary(serializedData);
      } else {
        serialized = JSON.stringify(serializedData, null, 2);
      }

      // Save to file or return data
      if (path) {
        await this._saveToFile(serialized, path);
        logger.info('Model saved successfully', { path, size: serialized.length });
        return path;
      } else {
        logger.info('Model serialized to memory', { size: serialized.length });
        return serialized;
      }

    } catch (error) {
      logger.error('Model save failed', { error: error.message, stack: error.stack });
      throw new Error(`Failed to save model: ${error.message}`);
    }
  }

  /**
   * Load model state dictionary - PyTorch compatible
   * @param {string|Object} data - File path or serialized data
   * @param {nn.Module} model - Model to load state into (optional for inspection)
   * @param {Object} options - Load options
   * @returns {Promise<Object>} Loaded state dictionary and metadata
   */
  async load(data, model = null, options = {}) {
    try {
      logger.info('Starting model deserialization', { hasModel: !!model, options });

      // Load serialized data
      let serializedData;
      if (typeof data === 'string') {
        if (data.startsWith('{') || data.startsWith('[')) {
          // JSON string
          serializedData = JSON.parse(data);
        } else {
          // File path
          serializedData = await this._loadFromFile(data);
        }
      } else {
        serializedData = data;
      }

      // Validate serialized data
      if (this.options.validateOnLoad) {
        this._validateSerializedData(serializedData);
      }

      // Extract state dictionary
      const stateDict = serializedData.model_state_dict;
      
      // Load state into model if provided
      if (model) {
        await this._loadStateDict(model, stateDict, options);
        logger.info('Model state loaded successfully', { 
          parameters: Object.keys(stateDict).length 
        });
      }

      // Return complete data for inspection
      const result = {
        state_dict: stateDict,
        metadata: serializedData.metadata,
        epoch: serializedData.epoch,
        loss: serializedData.loss,
        optimizer_state_dict: serializedData.optimizer_state_dict,
        version: serializedData.version,
        timestamp: serializedData.timestamp,
        pytorch_compatible: serializedData.pytorch_compatible
      };

      logger.info('Model loaded successfully', { 
        version: result.version,
        parameters: Object.keys(stateDict).length 
      });

      return result;

    } catch (error) {
      logger.error('Model load failed', { error: error.message, stack: error.stack });
      throw new Error(`Failed to load model: ${error.message}`);
    }
  }

  /**
   * Save training checkpoint with model and optimizer state
   * @param {Object} checkpoint - Checkpoint data
   * @param {string} path - File path
   * @returns {Promise<string>} File path
   */
  async saveCheckpoint(checkpoint, path) {
    const { model, optimizer, epoch, loss, ...metadata } = checkpoint;
    
    const checkpointData = {
      epoch,
      loss,
      ...metadata
    };

    if (model) {
      checkpointData.model = model;
    }

    if (optimizer) {
      checkpointData.optimizer_state = optimizer;
    }

    return this.save(model, path, checkpointData);
  }

  /**
   * Load training checkpoint
   * @param {string} path - File path
   * @param {nn.Module} model - Model to load into
   * @param {Optimizer} optimizer - Optimizer to load into
   * @returns {Promise<Object>} Checkpoint data
   */
  async loadCheckpoint(path, model = null, optimizer = null) {
    const data = await this.load(path, model);
    
    // Load optimizer state if provided
    if (optimizer && data.optimizer_state_dict) {
      await this._loadOptimizerState(optimizer, data.optimizer_state_dict);
    }

    return {
      epoch: data.epoch,
      loss: data.loss,
      model_state: data.state_dict,
      optimizer_state: data.optimizer_state_dict,
      metadata: data.metadata
    };
  }

  // Private methods

  async _extractStateDict(model) {
    const stateDict = {};
    
    if (!model || typeof model.parameters !== 'function') {
      throw new Error('Invalid model: must have parameters() method');
    }

    // Get all parameters
    const parameters = model.parameters();
    
    // Extract parameters with their names
    if (model._parameters) {
      // Direct parameter access
      for (const [name, param] of Object.entries(model._parameters)) {
        stateDict[name] = await this._serializeTensor(param);
      }
    }

    // Extract module parameters recursively
    if (model._modules) {
      for (const [moduleName, module] of Object.entries(model._modules)) {
        const moduleStateDict = await this._extractStateDict(module);
        for (const [paramName, paramData] of Object.entries(moduleStateDict)) {
          stateDict[`${moduleName}.${paramName}`] = paramData;
        }
      }
    }

    // Fallback: extract from parameters array with generated names
    if (Object.keys(stateDict).length === 0 && parameters.length > 0) {
      for (let i = 0; i < parameters.length; i++) {
        const param = parameters[i];
        const name = param._name || `param_${i}`;
        stateDict[name] = await this._serializeTensor(param);
      }
    }

    return stateDict;
  }

  async _serializeTensor(tensor) {
    if (!tensor) return null;

    const serialized = {
      shape: tensor.shape || [],
      dtype: tensor.dtype || 'float32',
      device: tensor.device || 'cpu',
      requires_grad: tensor.requires_grad || false,
      data: null
    };

    // Serialize tensor data
    if (tensor.data) {
      if (Array.isArray(tensor.data)) {
        serialized.data = tensor.data;
      } else if (tensor.data.constructor.name === 'Float32Array' || 
                 tensor.data.constructor.name === 'Int32Array') {
        serialized.data = Array.from(tensor.data);
      } else {
        // WebGPU tensor - need to read from GPU
        try {
          const data = await tensor.cpu(); // Move to CPU if needed
          serialized.data = Array.isArray(data.data) ? data.data : Array.from(data.data);
        } catch (error) {
          logger.warn('Failed to serialize WebGPU tensor, using fallback', { error: error.message });
          serialized.data = tensor.data ? Array.from(tensor.data) : [];
        }
      }
    }

    return serialized;
  }

  async _deserializeTensor(serialized, targetDevice = 'cpu') {
    if (!serialized) return null;

    // Use the global greed instance to create tensor
    const greed = globalThis.greed;
    if (!greed) {
      throw new Error('GreedJS not available for tensor deserialization');
    }

    // Create tensor from data
    const tensor = greed.torch.tensor(serialized.data, {
      dtype: serialized.dtype,
      device: targetDevice,
      requires_grad: serialized.requires_grad
    });

    // Reshape if necessary
    if (serialized.shape && serialized.shape.length > 1) {
      return tensor.view(serialized.shape);
    }

    return tensor;
  }

  async _loadStateDict(model, stateDict, options = {}) {
    const strict = options.strict !== false; // Default to strict mode
    const targetDevice = options.device || 'cpu';
    
    const modelParams = {};
    
    // Build model parameter mapping
    if (model._parameters) {
      Object.assign(modelParams, model._parameters);
    }

    // Add module parameters
    if (model._modules) {
      for (const [moduleName, module] of Object.entries(model._modules)) {
        if (module._parameters) {
          for (const [paramName, param] of Object.entries(module._parameters)) {
            modelParams[`${moduleName}.${paramName}`] = param;
          }
        }
      }
    }

    // Load parameters
    const loadedKeys = [];
    const missingKeys = [];
    const unexpectedKeys = [];

    for (const [key, paramData] of Object.entries(stateDict)) {
      if (key in modelParams) {
        const tensor = await this._deserializeTensor(paramData, targetDevice);
        // Copy data to existing parameter
        if (modelParams[key].data) {
          modelParams[key].data.set ? 
            modelParams[key].data.set(tensor.data) : 
            (modelParams[key].data = tensor.data);
        }
        loadedKeys.push(key);
      } else {
        unexpectedKeys.push(key);
      }
    }

    // Check for missing keys
    for (const key of Object.keys(modelParams)) {
      if (!loadedKeys.includes(key)) {
        missingKeys.push(key);
      }
    }

    if (strict && (missingKeys.length > 0 || unexpectedKeys.length > 0)) {
      const error = `State dict mismatch:\nMissing keys: ${missingKeys}\nUnexpected keys: ${unexpectedKeys}`;
      throw new Error(error);
    }

    logger.info('State dict loaded', {
      loaded: loadedKeys.length,
      missing: missingKeys.length,
      unexpected: unexpectedKeys.length
    });

    return { loadedKeys, missingKeys, unexpectedKeys };
  }

  async _extractOptimizerState(optimizer) {
    if (!optimizer || !optimizer.state_dict) {
      return null;
    }

    return {
      state: optimizer.state_dict(),
      param_groups: optimizer.param_groups
    };
  }

  async _loadOptimizerState(optimizer, state) {
    if (!optimizer || !state || !optimizer.load_state_dict) {
      return;
    }

    optimizer.load_state_dict(state.state);
    if (state.param_groups) {
      optimizer.param_groups = state.param_groups;
    }
  }

  _extractMetadata(model) {
    return {
      model_class: model.constructor.name,
      total_parameters: model.parameters ? model.parameters().length : 0,
      training_mode: model.training || false,
      device_type: this._detectPrimaryDevice(model),
      architecture: this._analyzeArchitecture(model)
    };
  }

  _detectPrimaryDevice(model) {
    if (!model.parameters) return 'cpu';
    
    const params = model.parameters();
    if (params.length === 0) return 'cpu';
    
    const firstParam = params[0];
    return firstParam.device || 'cpu';
  }

  _analyzeArchitecture(model) {
    const architecture = {
      modules: [],
      total_layers: 0,
      parameter_count: 0
    };

    if (model._modules) {
      for (const [name, module] of Object.entries(model._modules)) {
        architecture.modules.push({
          name,
          type: module.constructor.name,
          parameters: module.parameters ? module.parameters().length : 0
        });
        architecture.total_layers++;
      }
    }

    if (model.parameters) {
      architecture.parameter_count = model.parameters().reduce((sum, p) => {
        return sum + (p.data ? p.data.length || 0 : 0);
      }, 0);
    }

    return architecture;
  }

  _validateSerializedData(data) {
    if (!data.model_state_dict) {
      throw new Error('Invalid serialized data: missing model_state_dict');
    }

    if (!data.version) {
      logger.warn('Serialized data missing version information');
    }

    if (data.version && data.version !== this.version) {
      logger.warn('Version mismatch', { 
        expected: this.version, 
        found: data.version 
      });
    }
  }

  async _serializeToBinary(data) {
    // For now, use JSON serialization
    // In production, implement binary format for better performance
    return JSON.stringify(data);
  }

  async _saveToFile(data, path) {
    // Browser environment - use download
    if (typeof window !== 'undefined') {
      const blob = new Blob([data], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = path.split('/').pop() || 'model.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      return;
    }

    throw new Error('File saving not supported in current environment - use browser download');
  }

  async _loadFromFile(path) {
    // Browser environment - use FileReader API or fetch for URLs
    if (typeof window !== 'undefined') {
      if (path.startsWith('http://') || path.startsWith('https://') || path.startsWith('/')) {
        // URL - use fetch
        try {
          const response = await fetch(path);
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
          const text = await response.text();
          return JSON.parse(text);
        } catch (error) {
          throw new Error(`Failed to fetch model from ${path}: ${error.message}`);
        }
      } else {
        // File path - requires user input
        throw new Error('Local file loading in browser requires user file selection. Use createFileInput() method.');
      }
    }

    throw new Error('File loading not supported in current environment');
  }

  /**
   * Create file input for loading models in browser
   * @param {Function} callback - Callback function to handle loaded data
   * @returns {HTMLInputElement} File input element
   */
  createFileInput(callback) {
    if (typeof window === 'undefined') {
      throw new Error('File input only available in browser environment');
    }

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,.greed';
    input.style.display = 'none';

    input.addEventListener('change', async (event) => {
      const file = event.target.files[0];
      if (!file) return;

      try {
        const text = await file.text();
        const data = JSON.parse(text);
        callback(null, data);
      } catch (error) {
        callback(error, null);
      }
    });

    return input;
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    if (this.memoryManager) {
      await this.memoryManager.cleanup();
    }
  }
}

export default ModelSerializer;