module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
    jest: true
  },
  extends: [
    'eslint:recommended'
  ],
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module'
  },
  rules: {
    'no-unused-vars': ['error', { 'argsIgnorePattern': '^_' }],
    'no-console': 'off', // We have proper logging in place
    'prefer-const': 'error',
    'no-var': 'error',
    'semi': ['error', 'always'],
    'quotes': ['error', 'single', { 'allowTemplateLiterals': true }]
  },
  globals: {
    'GPUBufferUsage': 'readonly',
    'GPUShaderStage': 'readonly',
    'navigator': 'readonly',
    'performance': 'readonly',
    'window': 'readonly'
  }
};