import { useEffect, useRef } from 'react';
import { MNIST_SIZE, toMnistPixels } from './preprocess';

const SIZE = 280;
const BRUSH = 18;

type Props = {
  /** Called after each stroke with the drawing as 28×28 pixels (0–255). */
  onDraw: (pixels: number[]) => void;
  onClear: () => void;
};

/** White-on-black drawing area, like the MNIST images. */
export function DrawingCanvas({ onDraw, onClear }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const last = useRef<{ x: number; y: number } | null>(null);

  const context = () =>
    canvasRef.current?.getContext('2d', { willReadFrequently: true }) ?? null;

  const clear = () => {
    const ctx = context();
    if (!ctx) return;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, SIZE, SIZE);
  };

  useEffect(clear, []);

  const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) * SIZE) / rect.width,
      y: ((e.clientY - rect.top) * SIZE) / rect.height,
    };
  };

  const strokeTo = (to: { x: number; y: number }) => {
    const ctx = context();
    if (!ctx || !last.current) return;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = BRUSH;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(last.current.x, last.current.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    last.current = to;
  };

  const handleDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = point(e);
    last.current = p;
    const ctx = context();
    if (!ctx) return;
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(p.x, p.y, BRUSH / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const handleMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (last.current) strokeTo(point(e));
  };

  const handleUp = () => {
    if (!last.current) return;
    last.current = null;
    const ctx = context();
    if (!ctx) return;
    const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
    const gray = new Uint8ClampedArray(SIZE * SIZE);
    for (let i = 0; i < gray.length; i++) gray[i] = data[i * 4]; // red channel
    onDraw(toMnistPixels(gray, SIZE, SIZE));
  };

  return (
    <div className='drawing'>
      <canvas
        ref={canvasRef}
        className='drawing-canvas'
        width={SIZE}
        height={SIZE}
        aria-label='Draw a digit'
        onPointerDown={handleDown}
        onPointerMove={handleMove}
        onPointerUp={handleUp}
        onPointerCancel={handleUp}
      />
      <button
        onClick={() => {
          clear();
          onClear();
        }}
      >
        Clear
      </button>
    </div>
  );
}

/** Shows 28×28 pixels enlarged, so you can see what the network sees. */
export function PixelPreview({ pixels }: { pixels: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    const image = ctx.createImageData(MNIST_SIZE, MNIST_SIZE);
    pixels.forEach((v, i) => {
      image.data.set([v, v, v, 255], i * 4);
    });
    ctx.putImageData(image, 0, 0);
  }, [pixels]);
  return (
    <canvas
      ref={canvasRef}
      className='pixel-preview'
      width={MNIST_SIZE}
      height={MNIST_SIZE}
      aria-label='What the network sees'
    />
  );
}
