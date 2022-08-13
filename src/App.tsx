import './App.css';
import { Network, Neuron } from './neural-network/neural-network';
import { useState } from 'react';

export const network = new Network()
network.initializeSample(4);
console.log(network.layers[3][0]);

function App() {
  const [outputUI, setOutputUI] = useState(network.layers[3])
  const propagate = () => {
    network.propagateForward();
    network.propagateBack();
    setOutputUI(network.layers[3].map(o => o));
    console.log(network.layers[3][0]);
  };
  return (
    <div className='image-classification'>
      <h1>Image classification</h1>
      <h3>Output | Cost | Expected output</h3>
      <ul>
        {outputUI.map((output, i) => 
          <li key={i} >{i} --- {output.activation.toFixed(10)} | {(output as Neuron).cost?.toFixed(10)} | {(output as Neuron).expectedOutput}</li>
        )}
      </ul>
      <button onClick={propagate}>Propagate</button>
    </div>
  );
}

export default App;