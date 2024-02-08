declare global {
    interface Window { webkitAudioContext: typeof AudioContext }
}

import * as tf from '@tensorflow/tfjs';
import * as ort from 'onnxruntime-web';
import { setFeatureExtractor, process } from 'onnx-audio-processor';



async function initializeTensorFlow(): Promise<void> {
    await tf.ready();
}

async function processAudio(): Promise<void> {
    const vggishModelURL = './audioset-vggish-3.onnx'

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    await initializeTensorFlow();

    const inputElement = document.getElementById('audioFileInput') as HTMLInputElement;
    if (!inputElement.files) {
        throw new Error('No files selected');
    }
    const selectedFile = inputElement.files[0];

    const audioBuffer = await readWavFile(selectedFile, audioCtx);

    setFeatureExtractor(vggishModelURL);
    const pprocOutput = await process(audioBuffer);
    if (!pprocOutput) {
        throw new Error('pprocOutput is null or undefined');
    }
    console.log(pprocOutput);

    // Load bg noise detector model
    const bgModelPath = './bg_noise_detection.onnx';
    const bgSession = await ort.InferenceSession.create(bgModelPath);

    // Run bg noise detector model
    const bgInputArray = pprocOutput.data instanceof Float32Array ? Array.from(new Float32Array(pprocOutput.data.buffer)) : [];
    const bgInputName = bgSession.inputNames[0];

    const bgInputTensor = new ort.Tensor('float32', bgInputArray, pprocOutput.dims);
    const bgInputs = { [bgInputName]: bgInputTensor };
    const bgOutput = await bgSession.run(bgInputs);
    if (!bgOutput) {
        throw new Error('bgOutput is null or undefined');
    }

    console.log("bgOutput")
    // Output from the classifier
    console.log(bgOutput);
        
}

const processAudioButton = document.getElementById('processAudioButton');
if (processAudioButton) {
    processAudioButton.addEventListener('click', processAudio);
} else {
    console.error("Element with id 'processAudioButton' not found");
}


function readWavFile(file: File, audioCtx: AudioContext): Promise<AudioBuffer> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            if (e.target) {
                audioCtx.decodeAudioData(e.target.result as ArrayBuffer, (buffer) => {
                    resolve(buffer);
                });
            } else {
                reject(new Error('FileReader event target is null'));
            }
        };
        reader.onerror = function (e) {
            reject(e);
        };
        reader.readAsArrayBuffer(file);
    });
}
