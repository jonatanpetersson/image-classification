import './App.css';
import { Network } from './neural-network/neural-network';
import { DrawingCanvas, PixelPreview } from './drawing/DrawingCanvas';
import { useState } from 'react';

export const network = new Network();
const blank = Array(28 * 28).fill(0);

function App() {
  const [outputUI, setOutputUI] = useState(network.outputLayer.map(o => o));
  const [sample, setSample] = useState(0);
  const [passes, setPasses] = useState(1);
  const [accuracy, setAccuracy] = useState<number[]>([]);
  const [drawnPixels, setDrawnPixels] = useState<number[]>(blank);
  const [drawingGuess, setDrawingGuess] = useState<number | null>(null);
  const handleSampleInput = (event: React.ChangeEvent<HTMLInputElement>) => setSample(parseInt(event.target.value) || 0);
  const handlePassesInput = (event: React.ChangeEvent<HTMLInputElement>) => setPasses(parseInt(event.target.value) || 0);
  const initializeSample = () => network.initializeSample(sample);
  const runEpoch = () => {
    console.time('Running epoch');
    const result = network.runEpoch();
    console.timeEnd('Running epoch');
    setAccuracy(prev => [...prev, result]);
  };

  const guess = () => {
    network.propagateForward();
    setOutputUI(network.outputLayer.map(o => o));
  };
  const train = () => {
    for (let i = 0; i < passes; i++) {
      network.propagateForward();
      network.propagateBack();
    }
    network.propagateForward();
    setOutputUI(network.outputLayer.map(o => o));
  };
  const guessDrawing = (pixels: number[]) => {
    setDrawnPixels(pixels);
    network.setInput(pixels);
    network.propagateForward();
    setDrawingGuess(network.guessedDigit());
    setOutputUI(network.outputLayer.map(o => o));
  };
  const clearDrawing = () => {
    setDrawnPixels(blank);
    setDrawingGuess(null);
  };
  return (
    <div className='image-classification'>
      <h1>Image classification</h1>

      <section className='draw-section'>
        <DrawingCanvas onDraw={guessDrawing} onClear={clearDrawing} />
        <div className='draw-result'>
          <p>The network sees</p>
          <PixelPreview pixels={drawnPixels} />
          <p>Guess</p>
          <p className='guess'>{drawingGuess ?? '–'}</p>
        </div>
      </section>
      {accuracy.length === 0 && (
        <p className='hint'>The network is untrained: run a few epochs before drawing.</p>
      )}

      <h3>Output | Cost | Expected output</h3>
      <ul>
        {outputUI.map((output, i) =>
          <li key={i} >{i} --- {output.activation.toFixed(10)} | {output.cost?.toFixed(10)} | {output.expectedOutput}</li>
        )}
      </ul>
      <br />
      <button onClick={runEpoch}>Run epoch ({network.batch.length} samples)</button>
      {accuracy.length > 0 && (
        <p>Accuracy per epoch: {accuracy.map(a => `${(a * 100).toFixed(1)}%`).join(', ')}</p>
      )}
      <p>Set sample</p>
      <input type="number" onChange={handleSampleInput} />
      <p>Set passes per sample</p>
      <input type="number" onChange={handlePassesInput} />
      <button onClick={initializeSample}>Initialize sample</button>
      <button onClick={train}>Train</button>
      <button onClick={guess}>Guess</button>
    </div>
  );
}

export default App;
