declare global {
    interface Window { webkitAudioContext: typeof AudioContext }
}

import * as tf from '@tensorflow/tfjs';
import * as ort from 'onnxruntime-web';
import { setFeatureExtractor, process, setClassifierPath, runClassifier } from 'onnx-audio-processor';


async function initializeTensorFlow(): Promise<void> {
    await tf.ready();
}

async function processAudio(): Promise<void> {
    const processStartTime = performance.now();
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
    console.log("pprocOutput")
    console.log(pprocOutput);

    // Load bg noise detector model
    const bgModelPath = './bg_noise_detection.onnx';
    setClassifierPath(bgModelPath)
    runClassifier(pprocOutput);
    const processEndTime = performance.now();

    console.log(`Time taken for overall processing: ${processEndTime - processStartTime} milliseconds`);
        
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
