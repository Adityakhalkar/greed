/**
 * DataLoader - PyTorch-compatible data loading and preprocessing utilities
 * Supports batch processing, shuffling, and various data formats
 */
import logger from './logger.js';
import MemoryManager from './memory-manager.js';

class DataLoader {
  constructor(dataset, options = {}) {
    this.dataset = dataset;
    this.options = {
      batchSize: options.batchSize || 32,
      shuffle: options.shuffle !== false,
      dropLast: options.dropLast || false,
      numWorkers: options.numWorkers || 0, // Web workers for parallel loading
      pin_memory: options.pin_memory || false,
      timeout: options.timeout || 30000,
      collate_fn: options.collate_fn || this._defaultCollateFn.bind(this),
      ...options
    };
    
    this.memoryManager = new MemoryManager();
    this.currentEpoch = 0;
    this.currentBatch = 0;
    this.indices = [];
    this._createIndices();
  }

  /**
   * Get total number of batches
   */
  get length() {
    const totalSamples = this.dataset.length || this.dataset.size || 0;
    if (this.options.dropLast) {
      return Math.floor(totalSamples / this.options.batchSize);
    } else {
      return Math.ceil(totalSamples / this.options.batchSize);
    }
  }

  /**
   * Iterator interface - PyTorch compatible
   */
  [Symbol.iterator]() {
    return this;
  }

  /**
   * Next batch - iterator interface
   */
  async next() {
    if (this.currentBatch >= this.length) {
      // End of epoch
      this.currentEpoch++;
      this.currentBatch = 0;
      this._createIndices();
      return { done: true, value: undefined };
    }

    const batch = await this._getBatch(this.currentBatch);
    this.currentBatch++;
    
    return { done: false, value: batch };
  }

  /**
   * Reset iterator to beginning
   */
  reset() {
    this.currentBatch = 0;
    this._createIndices();
  }

  /**
   * Get a specific batch by index
   */
  async getBatch(batchIndex) {
    if (batchIndex >= this.length) {
      throw new Error(`Batch index ${batchIndex} out of range (0-${this.length - 1})`);
    }
    return await this._getBatch(batchIndex);
  }

  /**
   * Load all data as a single batch
   */
  async loadAll() {
    const batches = [];
    for (let i = 0; i < this.length; i++) {
      batches.push(await this._getBatch(i));
    }
    return batches;
  }

  /**
   * Get dataset statistics
   */
  getStats() {
    return {
      datasetSize: this.dataset.length || this.dataset.size || 0,
      batchSize: this.options.batchSize,
      numBatches: this.length,
      shuffle: this.options.shuffle,
      dropLast: this.options.dropLast,
      currentEpoch: this.currentEpoch,
      currentBatch: this.currentBatch
    };
  }

  // Private methods

  _createIndices() {
    const size = this.dataset.length || this.dataset.size || 0;
    this.indices = Array.from({ length: size }, (_, i) => i);
    
    if (this.options.shuffle) {
      this._shuffleArray(this.indices);
    }
  }

  _shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }

  async _getBatch(batchIndex) {
    const startIdx = batchIndex * this.options.batchSize;
    const endIdx = Math.min(startIdx + this.options.batchSize, this.indices.length);
    
    if (this.options.dropLast && (endIdx - startIdx) < this.options.batchSize) {
      throw new Error(`Incomplete batch dropped (drop_last=True)`);
    }

    const batchIndices = this.indices.slice(startIdx, endIdx);
    const samples = [];

    // Load samples
    for (const idx of batchIndices) {
      try {
        const sample = await this._getSample(idx);
        samples.push(sample);
      } catch (error) {
        logger.error('Sample loading failed', { index: idx, error: error.message });
        throw new Error(`Failed to load sample at index ${idx}: ${error.message}`);
      }
    }

    // Collate samples into batch
    return this.options.collate_fn(samples);
  }

  async _getSample(index) {
    if (typeof this.dataset.getItem === 'function') {
      return await this.dataset.getItem(index);
    } else if (typeof this.dataset.__getitem__ === 'function') {
      return await this.dataset.__getitem__(index);
    } else if (Array.isArray(this.dataset)) {
      return this.dataset[index];
    } else if (this.dataset[index] !== undefined) {
      return this.dataset[index];
    } else {
      throw new Error(`Cannot access sample at index ${index}`);
    }
  }

  _defaultCollateFn(samples) {
    if (!samples || samples.length === 0) {
      throw new Error('Cannot collate empty samples');
    }

    const firstSample = samples[0];
    
    // Handle different sample types
    if (Array.isArray(firstSample)) {
      // Array of features/labels pairs
      return this._collateArraySamples(samples);
    } else if (typeof firstSample === 'object' && firstSample !== null) {
      if (firstSample.data !== undefined || firstSample.shape !== undefined) {
        // Tensor-like objects
        return this._collateTensorSamples(samples);
      } else {
        // Dictionary-like objects
        return this._collateDictSamples(samples);
      }
    } else {
      // Primitive types
      return samples;
    }
  }

  _collateArraySamples(samples) {
    if (samples[0].length !== 2) {
      throw new Error('Array samples must have exactly 2 elements [features, labels]');
    }

    const features = [];
    const labels = [];

    for (const [feature, label] of samples) {
      features.push(feature);
      labels.push(label);
    }

    return [this._stackTensors(features), this._stackTensors(labels)];
  }

  _collateTensorSamples(samples) {
    return this._stackTensors(samples);
  }

  _collateDictSamples(samples) {
    const keys = Object.keys(samples[0]);
    const batched = {};

    for (const key of keys) {
      const values = samples.map(sample => sample[key]);
      batched[key] = this._stackTensors(values);
    }

    return batched;
  }

  _stackTensors(tensors) {
    if (!tensors || tensors.length === 0) {
      throw new Error('Cannot stack empty tensor list');
    }

    const greed = globalThis.greed;
    if (!greed) {
      throw new Error('GreedJS not available for tensor operations');
    }

    try {
      // Use torch.stack for tensor stacking
      if (greed.torch && greed.torch.stack) {
        return greed.torch.stack(tensors);
      }
      
      // Fallback: create tensor from array of data
      const data = tensors.map(t => {
        if (t && t.data) return Array.isArray(t.data) ? t.data : Array.from(t.data);
        if (Array.isArray(t)) return t;
        return [t];
      });
      
      return greed.torch.tensor(data);
    } catch (error) {
      logger.warn('Tensor stacking failed, returning array', { error: error.message });
      return tensors;
    }
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

/**
 * TensorDataset - Simple dataset from tensors
 * PyTorch-compatible tensor dataset
 */
class TensorDataset {
  constructor(...tensors) {
    if (tensors.length === 0) {
      throw new Error('At least one tensor is required');
    }

    // Validate all tensors have same first dimension
    const firstSize = tensors[0].shape ? tensors[0].shape[0] : tensors[0].length;
    for (let i = 1; i < tensors.length; i++) {
      const size = tensors[i].shape ? tensors[i].shape[0] : tensors[i].length;
      if (size !== firstSize) {
        throw new Error(`Size mismatch: tensor 0 has size ${firstSize} but tensor ${i} has size ${size}`);
      }
    }

    this.tensors = tensors;
    this.length = firstSize;
  }

  async getItem(index) {
    if (index >= this.length || index < 0) {
      throw new Error(`Index ${index} out of range [0, ${this.length})`);
    }

    const items = this.tensors.map(tensor => {
      if (tensor.shape && tensor.shape.length > 1) {
        // Multi-dimensional tensor - slice first dimension
        return tensor.slice([index, index + 1]).squeeze(0);
      } else if (tensor.data) {
        // 1D tensor or tensor-like object
        return tensor.data[index];
      } else if (Array.isArray(tensor)) {
        // Array
        return tensor[index];
      } else {
        throw new Error(`Cannot index tensor of type ${typeof tensor}`);
      }
    });

    return items.length === 1 ? items[0] : items;
  }

  get size() {
    return this.length;
  }
}

/**
 * ArrayDataset - Dataset from JavaScript arrays
 * Simple dataset wrapper for array data
 */
class ArrayDataset {
  constructor(data, labels = null) {
    this.data = data;
    this.labels = labels;
    this.length = Array.isArray(data) ? data.length : data.size || 0;

    if (labels && labels.length !== this.length) {
      throw new Error(`Data and labels length mismatch: ${this.length} vs ${labels.length}`);
    }
  }

  async getItem(index) {
    if (index >= this.length || index < 0) {
      throw new Error(`Index ${index} out of range [0, ${this.length})`);
    }

    const sample = this.data[index];
    if (this.labels) {
      return [sample, this.labels[index]];
    } else {
      return sample;
    }
  }

  get size() {
    return this.length;
  }
}

/**
 * CSVDataset - Load data from CSV format
 * Supports various CSV parsing options
 */
class CSVDataset {
  constructor(csvData, options = {}) {
    this.options = {
      hasHeader: options.hasHeader !== false,
      delimiter: options.delimiter || ',',
      targetColumn: options.targetColumn || -1, // Last column as target
      featureColumns: options.featureColumns || null, // All except target
      skipColumns: options.skipColumns || [],
      dtype: options.dtype || 'float32',
      ...options
    };

    this.data = [];
    this.labels = [];
    this.headers = [];
    this._parseCSV(csvData);
  }

  _parseCSV(csvData) {
    const lines = csvData.trim().split('\n');
    let dataLines = lines;

    if (this.options.hasHeader) {
      this.headers = lines[0].split(this.options.delimiter);
      dataLines = lines.slice(1);
    }

    for (const line of dataLines) {
      if (!line.trim()) continue;

      const values = line.split(this.options.delimiter).map(v => v.trim());
      
      // Extract target
      let target = null;
      if (this.options.targetColumn >= 0) {
        target = this._parseValue(values[this.options.targetColumn]);
      } else if (this.options.targetColumn < 0) {
        target = this._parseValue(values[values.length + this.options.targetColumn]);
      }

      // Extract features
      const features = [];
      if (this.options.featureColumns) {
        for (const col of this.options.featureColumns) {
          if (!this.options.skipColumns.includes(col)) {
            features.push(this._parseValue(values[col]));
          }
        }
      } else {
        for (let i = 0; i < values.length; i++) {
          if (i !== this.options.targetColumn && 
              i !== (values.length + this.options.targetColumn) &&
              !this.options.skipColumns.includes(i)) {
            features.push(this._parseValue(values[i]));
          }
        }
      }

      this.data.push(features);
      if (target !== null) {
        this.labels.push(target);
      }
    }

    this.length = this.data.length;
  }

  _parseValue(str) {
    const num = parseFloat(str);
    return isNaN(num) ? str : num;
  }

  async getItem(index) {
    if (index >= this.length || index < 0) {
      throw new Error(`Index ${index} out of range [0, ${this.length})`);
    }

    const features = this.data[index];
    if (this.labels.length > 0) {
      return [features, this.labels[index]];
    } else {
      return features;
    }
  }

  get size() {
    return this.length;
  }

  getFeatureNames() {
    if (this.headers.length === 0) return null;
    
    const featureNames = [];
    for (let i = 0; i < this.headers.length; i++) {
      if (i !== this.options.targetColumn && 
          i !== (this.headers.length + this.options.targetColumn) &&
          !this.options.skipColumns.includes(i)) {
        featureNames.push(this.headers[i]);
      }
    }
    return featureNames;
  }
}

/**
 * DataPreprocessor - Data preprocessing utilities
 * Normalization, scaling, and transformation utilities
 */
class DataPreprocessor {
  constructor() {
    this.stats = {};
    this.transformations = [];
  }

  /**
   * Fit preprocessing parameters on training data
   */
  fit(data, options = {}) {
    const { method = 'standardize', axis = null } = options;
    
    if (!Array.isArray(data) && (!data.data || !data.shape)) {
      throw new Error('Data must be array or tensor');
    }

    const tensorData = data.data || data;
    
    switch (method) {
      case 'standardize':
        this._fitStandardization(tensorData, axis);
        break;
      case 'normalize':
        this._fitNormalization(tensorData, axis);
        break;
      case 'minmax':
        this._fitMinMaxScaling(tensorData, axis);
        break;
      default:
        throw new Error(`Unknown preprocessing method: ${method}`);
    }

    this.stats.method = method;
    this.stats.axis = axis;
    return this;
  }

  /**
   * Transform data using fitted parameters
   */
  transform(data) {
    if (!this.stats.method) {
      throw new Error('Preprocessor not fitted. Call fit() first.');
    }

    const greed = globalThis.greed;
    if (!greed) {
      throw new Error('GreedJS not available');
    }

    switch (this.stats.method) {
      case 'standardize':
        return this._applyStandardization(data);
      case 'normalize':
        return this._applyNormalization(data);
      case 'minmax':
        return this._applyMinMaxScaling(data);
      default:
        return data;
    }
  }

  /**
   * Fit and transform in one step
   */
  fitTransform(data, options = {}) {
    return this.fit(data, options).transform(data);
  }

  _fitStandardization(data, axis) {
    // Calculate mean and standard deviation
    const flatData = Array.isArray(data) ? data.flat() : Array.from(data);
    
    const mean = flatData.reduce((sum, val) => sum + val, 0) / flatData.length;
    const variance = flatData.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / flatData.length;
    const std = Math.sqrt(variance + 1e-8); // Small epsilon to avoid division by zero

    this.stats.mean = mean;
    this.stats.std = std;
  }

  _fitNormalization(data, axis) {
    const flatData = Array.isArray(data) ? data.flat() : Array.from(data);
    const magnitude = Math.sqrt(flatData.reduce((sum, val) => sum + val * val, 0));
    
    this.stats.magnitude = magnitude + 1e-8;
  }

  _fitMinMaxScaling(data, axis) {
    const flatData = Array.isArray(data) ? data.flat() : Array.from(data);
    
    const min = Math.min(...flatData);
    const max = Math.max(...flatData);
    
    this.stats.min = min;
    this.stats.max = max;
    this.stats.range = max - min + 1e-8;
  }

  _applyStandardization(data) {
    const greed = globalThis.greed;
    
    if (data.data && data.shape) {
      // Tensor
      const meanTensor = greed.torch.tensor([this.stats.mean]);
      const stdTensor = greed.torch.tensor([this.stats.std]);
      return data.sub(meanTensor).div(stdTensor);
    } else {
      // Array
      return data.map(val => (val - this.stats.mean) / this.stats.std);
    }
  }

  _applyNormalization(data) {
    const greed = globalThis.greed;
    
    if (data.data && data.shape) {
      // Tensor  
      const magTensor = greed.torch.tensor([this.stats.magnitude]);
      return data.div(magTensor);
    } else {
      // Array
      return data.map(val => val / this.stats.magnitude);
    }
  }

  _applyMinMaxScaling(data) {
    const greed = globalThis.greed;
    
    if (data.data && data.shape) {
      // Tensor
      const minTensor = greed.torch.tensor([this.stats.min]);
      const rangeTensor = greed.torch.tensor([this.stats.range]);
      return data.sub(minTensor).div(rangeTensor);
    } else {
      // Array
      return data.map(val => (val - this.stats.min) / this.stats.range);
    }
  }

  getStats() {
    return { ...this.stats };
  }
}

export { DataLoader, TensorDataset, ArrayDataset, CSVDataset, DataPreprocessor };
export default DataLoader;