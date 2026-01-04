import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  Zap, 
  Activity, 
  Brain, 
  Sparkles,
  Download,
  Trash2,
  Eye,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Play,
  Image as ImageIcon
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { 
  checkHealth, 
  predictAction, 
  annotateImage, 
  PredictionResult
} from './api';

// Particle Background Component
const ParticleBackground = () => {
  const particles = Array.from({ length: 50 }, (_, i) => ({
    id: i,
    x: Math.random() * 100,
    y: Math.random() * 100,
    duration: 5 + Math.random() * 10,
    delay: Math.random() * 5,
  }));

  return (
    <div className="particles">
      {particles.map((p) => (
        <motion.div
          key={p.id}
          className="particle"
          style={{ left: `${p.x}%`, top: `${p.y}%` }}
          animate={{
            y: [0, -30, 0],
            opacity: [0.2, 0.8, 0.2],
          }}
          transition={{
            duration: p.duration,
            delay: p.delay,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
};

// Status Indicator Component
const StatusIndicator = ({ isConnected }: { isConnected: boolean }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.8 }}
    animate={{ opacity: 1, scale: 1 }}
    className={`flex items-center gap-2 px-4 py-2 rounded-full glass ${
      isConnected ? 'border-green-500/50' : 'border-red-500/50'
    }`}
  >
    <motion.div
      className={`w-2 h-2 rounded-full ${
        isConnected ? 'bg-green-500' : 'bg-red-500'
      }`}
      animate={{
        scale: [1, 1.2, 1],
        opacity: [1, 0.7, 1],
      }}
      transition={{ duration: 2, repeat: Infinity }}
    />
    <span className="text-sm font-mono">
      {isConnected ? 'NEURAL LINK ACTIVE' : 'CONNECTING...'}
    </span>
  </motion.div>
);

// Upload Zone Component
const UploadZone = ({ 
  onFileSelect, 
  isProcessing 
}: { 
  onFileSelect: (file: File) => void;
  isProcessing: boolean;
}) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.jpg', '.png', '.webp'] },
    multiple: false,
    disabled: isProcessing,
  });

  const rootProps = getRootProps();

  return (
    <motion.div
      onClick={rootProps.onClick}
      onKeyDown={rootProps.onKeyDown}
      onFocus={rootProps.onFocus}
      onBlur={rootProps.onBlur}
      tabIndex={rootProps.tabIndex}
      role={rootProps.role}
      className={`relative cursor-pointer rounded-2xl p-8 transition-all duration-300 ${
        isDragActive
          ? 'gradient-border bg-cyber-primary/10'
          : 'glass hover:border-cyber-primary/50'
      } ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
      whileHover={{ scale: isProcessing ? 1 : 1.02 }}
      whileTap={{ scale: isProcessing ? 1 : 0.98 }}
      onDragEnter={rootProps.onDragEnter}
      onDragOver={rootProps.onDragOver}
      onDragLeave={rootProps.onDragLeave}
      onDrop={rootProps.onDrop}
    >
      <input {...getInputProps()} />
      
      <div className="flex flex-col items-center gap-6 py-8">
        <motion.div
          className="relative"
          animate={isDragActive ? { scale: 1.1, rotate: 5 } : {}}
        >
          <div className="w-24 h-24 rounded-2xl bg-gradient-to-br from-cyber-primary/20 to-cyber-secondary/20 flex items-center justify-center">
            <Upload className="w-10 h-10 text-cyber-primary" />
          </div>
          <motion.div
            className="absolute -top-2 -right-2 w-8 h-8 rounded-full bg-cyber-secondary/20 flex items-center justify-center"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Sparkles className="w-4 h-4 text-cyber-secondary" />
          </motion.div>
        </motion.div>

        <div className="text-center">
          <h3 className="font-cyber text-xl mb-2 text-white">
            {isDragActive ? 'DROP TO ANALYZE' : 'UPLOAD IMAGE'}
          </h3>
          <p className="text-gray-400 text-sm">
            Drag & drop or click to select • JPG, PNG, WebP
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-500">
          <ImageIcon className="w-4 h-4" />
          <span>AI-powered action recognition</span>
        </div>
      </div>

      {/* Animated border */}
      <div className="absolute inset-0 rounded-2xl overflow-hidden pointer-events-none">
        <motion.div
          className="absolute inset-0 opacity-30"
          style={{
            background: 'linear-gradient(90deg, transparent, #00f0ff, transparent)',
            transform: 'translateX(-100%)',
          }}
          animate={{ transform: ['translateX(-100%)', 'translateX(100%)'] }}
          transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
        />
      </div>
    </motion.div>
  );
};

// Image Preview Component
const ImagePreview = ({ 
  src, 
  onClear 
}: { 
  src: string; 
  onClear: () => void;
}) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    className="relative rounded-2xl overflow-hidden glass"
  >
    <img 
      src={src} 
      alt="Preview" 
      className="w-full h-64 object-contain bg-black/50"
    />
    <motion.button
      onClick={onClear}
      className="absolute top-3 right-3 p-2 rounded-full bg-red-500/20 border border-red-500/50 text-red-400 hover:bg-red-500/30 transition-colors"
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
    >
      <Trash2 className="w-4 h-4" />
    </motion.button>

    {/* Scan line effect */}
    <motion.div
      className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-cyber-primary to-transparent"
      initial={{ top: 0 }}
      animate={{ top: '100%' }}
      transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
    />
  </motion.div>
);

// Action Button Component
const ActionButton = ({ 
  onClick, 
  icon: Icon, 
  label, 
  variant = 'primary',
  disabled = false,
  loading = false,
}: { 
  onClick: () => void;
  icon: React.ElementType;
  label: string;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
  loading?: boolean;
}) => (
  <motion.button
    onClick={onClick}
    disabled={disabled || loading}
    className={`cyber-btn ${variant === 'primary' ? 'cyber-btn-primary' : 'cyber-btn-secondary'} 
      w-full flex items-center justify-center gap-3 rounded-xl disabled:opacity-50 disabled:cursor-not-allowed`}
    whileHover={{ scale: disabled ? 1 : 1.02 }}
    whileTap={{ scale: disabled ? 1 : 0.98 }}
  >
    {loading ? (
      <Loader2 className="w-5 h-5 animate-spin" />
    ) : (
      <Icon className="w-5 h-5" />
    )}
    <span>{loading ? 'PROCESSING...' : label}</span>
  </motion.button>
);

// Results Display Component
const ResultsDisplay = ({ 
  result, 
  annotatedImage 
}: { 
  result: PredictionResult;
  annotatedImage?: string;
}) => {
  const sortedPredictions = Object.entries(result.top5_predictions)
    .sort(([, a], [, b]) => b - a);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Main Prediction */}
      <motion.div
        className="relative glass-strong rounded-2xl p-6 overflow-hidden"
        initial={{ scale: 0.9 }}
        animate={{ scale: 1 }}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-cyber-primary/5 to-cyber-secondary/5" />
        
        <div className="relative text-center">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-1 rounded-full bg-cyber-primary/10 border border-cyber-primary/30 mb-4"
          >
            <CheckCircle2 className="w-4 h-4 text-cyber-primary" />
            <span className="text-sm font-mono text-cyber-primary">ANALYSIS COMPLETE</span>
          </motion.div>

          <motion.h2
            className="font-cyber text-3xl md:text-4xl font-bold mb-2"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <span className="text-gradient">{result.prediction}</span>
          </motion.h2>

          <motion.div
            className="flex items-center justify-center gap-2"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <Activity className="w-5 h-5 text-green-400" />
            <span className="text-2xl font-mono text-green-400">
              {(result.confidence * 100).toFixed(1)}%
            </span>
            <span className="text-gray-400 text-sm">confidence</span>
          </motion.div>
        </div>

        {/* Animated ring */}
        <motion.div
          className="absolute inset-0 rounded-2xl border-2 border-cyber-primary/20"
          animate={{
            boxShadow: [
              '0 0 0 0 rgba(0, 240, 255, 0)',
              '0 0 0 10px rgba(0, 240, 255, 0.1)',
              '0 0 0 0 rgba(0, 240, 255, 0)',
            ],
          }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      </motion.div>

      {/* Top 5 Predictions */}
      <div className="glass rounded-2xl p-6">
        <h3 className="font-cyber text-lg mb-4 flex items-center gap-2">
          <Brain className="w-5 h-5 text-cyber-secondary" />
          TOP PREDICTIONS
        </h3>

        <div className="space-y-3">
          {sortedPredictions.map(([action, probability], index) => (
            <motion.div
              key={action}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 * index }}
              className="relative"
            >
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-3">
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                    index === 0 
                      ? 'bg-cyber-primary/20 text-cyber-primary' 
                      : 'bg-gray-800 text-gray-400'
                  }`}>
                    {index + 1}
                  </span>
                  <span className={`font-mono ${index === 0 ? 'text-white' : 'text-gray-400'}`}>
                    {action}
                  </span>
                </div>
                <span className={`font-mono text-sm ${
                  index === 0 ? 'text-cyber-primary' : 'text-gray-500'
                }`}>
                  {(probability * 100).toFixed(1)}%
                </span>
              </div>
              
              <div className="cyber-progress">
                <motion.div
                  className="cyber-progress-bar"
                  initial={{ width: 0 }}
                  animate={{ width: `${probability * 100}%` }}
                  transition={{ delay: 0.2 + 0.1 * index, duration: 0.5 }}
                  style={{
                    background: index === 0 
                      ? 'linear-gradient(90deg, #00f0ff, #ff00e5)' 
                      : 'rgba(0, 240, 255, 0.3)',
                  }}
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Annotated Image */}
      {annotatedImage && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-2xl p-6"
        >
          <h3 className="font-cyber text-lg mb-4 flex items-center gap-2">
            <Eye className="w-5 h-5 text-cyber-accent" />
            ANNOTATED OUTPUT
          </h3>
          
          <div className="relative rounded-xl overflow-hidden">
            <img 
              src={`data:image/jpeg;base64,${annotatedImage}`} 
              alt="Annotated" 
              className="w-full"
            />
          </div>

          <motion.a
            href={`data:image/jpeg;base64,${annotatedImage}`}
            download="action_annotated.jpg"
            className="cyber-btn cyber-btn-secondary w-full flex items-center justify-center gap-3 rounded-xl mt-4"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Download className="w-5 h-5" />
            <span>DOWNLOAD IMAGE</span>
          </motion.a>
        </motion.div>
      )}
    </motion.div>
  );
};

// Error Display Component
const ErrorDisplay = ({ message }: { message: string }) => (
  <motion.div
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
    className="glass rounded-2xl p-6 border border-red-500/30"
  >
    <div className="flex items-center gap-3 text-red-400">
      <AlertCircle className="w-6 h-6" />
      <div>
        <h3 className="font-cyber text-lg">ERROR DETECTED</h3>
        <p className="text-sm text-red-400/70 mt-1">{message}</p>
      </div>
    </div>
  </motion.div>
);

// Loading Overlay Component
const LoadingOverlay = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50"
  >
    <motion.div
      className="text-center"
      initial={{ scale: 0.8 }}
      animate={{ scale: 1 }}
    >
      <div className="relative">
        <motion.div
          className="w-24 h-24 rounded-full border-4 border-cyber-primary/30"
          style={{ borderTopColor: '#00f0ff', borderRightColor: '#ff00e5' }}
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        />
        <Brain className="w-10 h-10 text-cyber-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
      </div>
      
      <motion.p
        className="font-cyber text-xl mt-6 text-cyber-primary"
        animate={{ opacity: [1, 0.5, 1] }}
        transition={{ duration: 1.5, repeat: Infinity }}
      >
        ANALYZING
      </motion.p>
      <p className="text-gray-400 text-sm mt-2 font-mono">
        Neural network processing...
      </p>
    </motion.div>
  </motion.div>
);

// Main App Component
export default function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkHealth().then(setIsConnected);
    const interval = setInterval(() => {
      checkHealth().then(setIsConnected);
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setAnnotatedImage(null);
    setError(null);
  }, []);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setAnnotatedImage(null);
    setError(null);
  }, []);

  const handlePredict = useCallback(async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      const response = await predictAction(selectedFile);
      if ('error' in response) {
        setError(response.error);
      } else {
        setResult(response);
        setAnnotatedImage(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  }, [selectedFile]);

  const handleAnnotate = useCallback(async () => {
    if (!selectedFile) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      const response = await annotateImage(selectedFile);
      if ('error' in response) {
        setError(response.error);
      } else {
        setResult(response);
        setAnnotatedImage(response.annotated_image);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  }, [selectedFile]);

  return (
    <div className="min-h-screen cyber-grid relative">
      <ParticleBackground />
      
      <AnimatePresence>
        {isProcessing && <LoadingOverlay />}
      </AnimatePresence>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <motion.div
            className="inline-flex items-center gap-3 mb-6"
            animate={{ y: [0, -5, 0] }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyber-primary to-cyber-secondary flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <h1 className="font-cyber text-4xl md:text-5xl font-bold">
              <span className="text-gradient">NEURAL</span>
              <span className="text-white"> VISION</span>
            </h1>
          </motion.div>
          
          <p className="text-gray-400 max-w-xl mx-auto mb-4">
            Advanced AI-powered action recognition using CNN-LSTM neural networks
            trained on UCF101 dataset with 101 action classes
          </p>

          <StatusIndicator isConnected={isConnected} />
        </motion.header>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left Column - Upload */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            <div className="glass rounded-2xl p-6">
              <h2 className="font-cyber text-xl mb-6 flex items-center gap-2">
                <Upload className="w-5 h-5 text-cyber-primary" />
                INPUT MODULE
              </h2>

              {!previewUrl ? (
                <UploadZone 
                  onFileSelect={handleFileSelect} 
                  isProcessing={isProcessing}
                />
              ) : (
                <div className="space-y-4">
                  <ImagePreview src={previewUrl} onClear={handleClear} />
                  
                  <div className="grid grid-cols-2 gap-4">
                    <ActionButton
                      onClick={handlePredict}
                      icon={Zap}
                      label="ANALYZE"
                      variant="primary"
                      disabled={!isConnected}
                      loading={isProcessing}
                    />
                    <ActionButton
                      onClick={handleAnnotate}
                      icon={Play}
                      label="ANNOTATE"
                      variant="secondary"
                      disabled={!isConnected}
                      loading={isProcessing}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Info Card */}
            <motion.div
              className="glass rounded-2xl p-6"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <h3 className="font-cyber text-lg mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-cyber-secondary" />
                SYSTEM INFO
              </h3>
              
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Model Architecture</span>
                  <span className="font-mono text-cyber-primary">CNN-LSTM</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Backbone</span>
                  <span className="font-mono text-cyber-primary">ResNet-50</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Dataset</span>
                  <span className="font-mono text-cyber-primary">UCF101</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Action Classes</span>
                  <span className="font-mono text-cyber-primary">101</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">GPU Acceleration</span>
                  <span className="font-mono text-green-400">NVIDIA T4</span>
                </div>
              </div>
            </motion.div>
          </motion.div>

          {/* Right Column - Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="glass rounded-2xl p-6 min-h-[400px]">
              <h2 className="font-cyber text-xl mb-6 flex items-center gap-2">
                <Activity className="w-5 h-5 text-cyber-secondary" />
                OUTPUT MODULE
              </h2>

              <AnimatePresence mode="wait">
                {error ? (
                  <ErrorDisplay key="error" message={error} />
                ) : result ? (
                  <ResultsDisplay 
                    key="results" 
                    result={result} 
                    annotatedImage={annotatedImage || undefined}
                  />
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-center justify-center h-80 text-center"
                  >
                    <motion.div
                      className="w-20 h-20 rounded-2xl bg-gray-800/50 flex items-center justify-center mb-4"
                      animate={{ rotate: [0, 5, -5, 0] }}
                      transition={{ duration: 4, repeat: Infinity }}
                    >
                      <Activity className="w-10 h-10 text-gray-600" />
                    </motion.div>
                    <p className="text-gray-500 font-cyber">AWAITING INPUT</p>
                    <p className="text-gray-600 text-sm mt-2">
                      Upload an image to begin analysis
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </div>

        {/* Footer */}
        <motion.footer
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="text-center mt-12 text-gray-500 text-sm"
        >
          <p className="font-mono">
            NEURAL VISION SYSTEM v1.0 • Powered by Modal.com & FastAPI
          </p>
        </motion.footer>
      </div>
    </div>
  );
}
