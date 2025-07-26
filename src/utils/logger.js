/**
 * Logger utility for Greed.js v2.0
 * Provides configurable logging with levels and environment-aware output
 */

class Logger {
  constructor(options = {}) {
    this.config = {
      level: options.level || (typeof process !== 'undefined' && process.env.NODE_ENV === 'development' ? 'debug' : 'warn'),
      enableConsole: options.enableConsole !== false,
      prefix: options.prefix || 'Greed',
      timestamp: options.timestamp !== false,
      ...options
    };

    // Log levels with priority (lower number = higher priority)
    this.levels = {
      error: 0,
      warn: 1,
      info: 2,
      debug: 3
    };

    this.currentLevelPriority = this.levels[this.config.level] || 1;
  }

  _shouldLog(level) {
    return this.levels[level] <= this.currentLevelPriority;
  }

  _formatMessage(level, message, ...args) {
    const timestamp = this.config.timestamp ? new Date().toISOString() : '';
    const prefix = this.config.prefix ? `[${this.config.prefix}]` : '';
    const levelStr = `[${level.toUpperCase()}]`;
    
    const parts = [timestamp, prefix, levelStr].filter(Boolean);
    return {
      formatted: `${parts.join(' ')} ${message}`,
      args
    };
  }

  error(message, ...args) {
    if (this._shouldLog('error') && this.config.enableConsole) {
      const { formatted, args: extraArgs } = this._formatMessage('error', message, ...args);
      console.error(formatted, ...extraArgs);
    }
  }

  warn(message, ...args) {
    if (this._shouldLog('warn') && this.config.enableConsole) {
      const { formatted, args: extraArgs } = this._formatMessage('warn', message, ...args);
      console.warn(formatted, ...extraArgs);
    }
  }

  info(message, ...args) {
    if (this._shouldLog('info') && this.config.enableConsole) {
      const { formatted, args: extraArgs } = this._formatMessage('info', message, ...args);
      console.log(formatted, ...extraArgs);
    }
  }

  debug(message, ...args) {
    if (this._shouldLog('debug') && this.config.enableConsole) {
      const { formatted, args: extraArgs } = this._formatMessage('debug', message, ...args);
      console.log(formatted, ...extraArgs);
    }
  }

  setLevel(level) {
    if (level in this.levels) {
      this.config.level = level;
      this.currentLevelPriority = this.levels[level];
    }
  }

  // Create child logger with same config but different prefix
  child(prefix) {
    return new Logger({
      ...this.config,
      prefix: `${this.config.prefix}:${prefix}`
    });
  }
}

// Create default logger instance
const defaultLogger = new Logger();

// Export both class and default instance
export default defaultLogger;
export { Logger };