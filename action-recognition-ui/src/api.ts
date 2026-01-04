export const API_URL = 'https://kooltaurion--action-recognition-api-fastapi-app.modal.run';

export interface PredictionResult {
  prediction: string;
  confidence: number;
  top5_predictions: Record<string, number>;
  all_classes_count?: number;
}

export interface AnnotationResult extends PredictionResult {
  annotated_image: string;
}

export interface ApiError {
  error: string;
  traceback?: string;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}

export async function predictAction(file: File): Promise<PredictionResult | ApiError> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
}

export async function annotateImage(file: File): Promise<AnnotationResult | ApiError> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_URL}/annotate`, {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
}

export async function getClasses(): Promise<{ classes: string[]; count: number }> {
  const response = await fetch(`${API_URL}/classes`);
  return response.json();
}
