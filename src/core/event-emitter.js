/**
 * EventEmitter - Base class for module communication
 * Provides event-driven architecture for decoupled module interaction
 */
class EventEmitter {
  constructor() {
    this._events = new Map();
    this._maxListeners = 10;
  }

  /**
   * Add event listener
   */
  on(event, listener) {
    if (typeof listener !== 'function') {
      throw new TypeError('Listener must be a function');
    }

    if (!this._events.has(event)) {
      this._events.set(event, []);
    }

    const listeners = this._events.get(event);
    listeners.push(listener);

    if (listeners.length > this._maxListeners) {
      this.emit('maxListenersExceeded', { event, count: listeners.length, limit: this._maxListeners });
    }

    return this;
  }

  /**
   * Add one-time event listener
   */
  once(event, listener) {
    const onceWrapper = (...args) => {
      listener.apply(this, args);
      this.off(event, onceWrapper);
    };
    return this.on(event, onceWrapper);
  }

  /**
   * Remove event listener
   */
  off(event, listener) {
    if (!this._events.has(event)) {
      return this;
    }

    const listeners = this._events.get(event);
    const index = listeners.indexOf(listener);
    
    if (index !== -1) {
      listeners.splice(index, 1);
    }

    if (listeners.length === 0) {
      this._events.delete(event);
    }

    return this;
  }

  /**
   * Emit event to all listeners
   */
  emit(event, ...args) {
    if (!this._events.has(event)) {
      return false;
    }

    const listeners = this._events.get(event).slice(); // Copy to avoid modification during iteration
    
    for (const listener of listeners) {
      try {
        listener.apply(this, args);
      } catch (error) {
        // Log error and emit error event for handling
        this.emit('error', error, event);
      }
    }

    return true;
  }

  /**
   * Remove all listeners for event or all events
   */
  removeAllListeners(event) {
    if (event) {
      this._events.delete(event);
    } else {
      this._events.clear();
    }
    return this;
  }

  /**
   * Get listener count for event
   */
  listenerCount(event) {
    return this._events.has(event) ? this._events.get(event).length : 0;
  }

  /**
   * Set maximum listeners per event
   */
  setMaxListeners(n) {
    if (typeof n !== 'number' || n < 0 || isNaN(n)) {
      throw new TypeError('n must be a non-negative number');
    }
    this._maxListeners = n;
    return this;
  }

  /**
   * Get event names
   */
  eventNames() {
    return Array.from(this._events.keys());
  }

  /**
   * Async event emission with error handling
   */
  async emitAsync(event, ...args) {
    if (!this._events.has(event)) {
      return [];
    }

    const listeners = this._events.get(event).slice();
    const promises = listeners.map(async (listener) => {
      try {
        return await listener.apply(this, args);
      } catch (error) {
        this.emit('error', error, event);
        throw error;
      }
    });

    return Promise.allSettled(promises);
  }
}

export default EventEmitter;