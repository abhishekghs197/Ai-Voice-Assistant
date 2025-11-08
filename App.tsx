import React, { useState, useRef, useCallback, useEffect } from 'react';
// FIX: Remove 'LiveSession' as it is not an exported member of '@google/genai'.
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from "@google/genai";

const SUPPORTED_LANGUAGES = [
  { name: 'English', displayName: 'English', voice: 'Zephyr' },
  { name: 'Bengali', displayName: 'বাংলা (Bengali)', voice: 'Kore' },
  { name: 'Hindi', displayName: 'हिन्दी (Hindi)', voice: 'Kore' },
  { name: 'Spanish', displayName: 'Español', voice: 'Puck' },
  { name: 'French', displayName: 'Français', voice: 'Charon' },
  { name: 'German', displayName: 'Deutsch', voice: 'Fenrir' },
  { name: 'Japanese', displayName: '日本語', voice: 'Puck' },
  { name: 'Mandarin Chinese', displayName: '中文', voice: 'Kore' },
  { name: 'Italian', displayName: 'Italiano', voice: 'Charon' },
  { name: 'Portuguese', displayName: 'Português', voice: 'Puck' },
  { name: 'Russian', displayName: 'Русский', voice: 'Fenrir' },
  { name: 'Arabic', displayName: 'العربية (Arabic)', voice: 'Zephyr' },
] as const;

type LanguageName = typeof SUPPORTED_LANGUAGES[number]['name'];
type SessionStatus = 'IDLE' | 'CONNECTING' | 'CONNECTED' | 'ERROR' | 'CLOSED';
type TranscriptionTurn = {
  speaker: 'user' | 'model';
  text: string;
  isFinal: boolean;
};

// --- Audio Utility Functions ---
function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createPcmBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// --- React Component ---
const App: React.FC = () => {
  const [status, setStatus] = useState<SessionStatus>('IDLE');
  const [transcription, setTranscription] = useState<TranscriptionTurn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<LanguageName>('English');
  const [isModelThinking, setIsModelThinking] = useState(false);

  // FIX: Update the ref type to use 'any' since 'LiveSession' is not an exported type.
  const sessionPromiseRef = useRef<Promise<any> | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  
  const currentInputTranscriptionRef = useRef('');
  const currentOutputTranscriptionRef = useRef('');
  const nextAudioStartTimeRef = useRef(0);
  const audioPlaybackQueueRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const transcriptionContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (transcriptionContainerRef.current) {
        transcriptionContainerRef.current.scrollTop = transcriptionContainerRef.current.scrollHeight;
    }
  }, [transcription, isModelThinking]);

  const updateTranscription = useCallback((newTurn: Partial<TranscriptionTurn>) => {
    setTranscription(prev => {
        const newTranscription = [...prev];
        const lastTurn = newTranscription[newTranscription.length - 1];
        
        if (lastTurn && lastTurn.speaker === newTurn.speaker && !lastTurn.isFinal) {
            lastTurn.text = newTurn.text ?? lastTurn.text;
            lastTurn.isFinal = newTurn.isFinal ?? lastTurn.isFinal;
        } else if(newTurn.speaker && newTurn.text) {
            newTranscription.push({ speaker: newTurn.speaker, text: newTurn.text, isFinal: newTurn.isFinal ?? false });
        }
        return newTranscription;
    });
  }, []);

  const stopConversation = useCallback(async () => {
    setStatus('CLOSED');
    setIsModelThinking(false);
    if (sessionPromiseRef.current) {
      try {
        const session = await sessionPromiseRef.current;
        session.close();
      } catch (e) {
        // Ignore errors on close, as the session might already be terminated.
      } finally {
        sessionPromiseRef.current = null;
      }
    }
    if (microphoneStreamRef.current) {
      microphoneStreamRef.current.getTracks().forEach(track => track.stop());
      microphoneStreamRef.current = null;
    }
    if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current = null;
    }
    if (inputAudioContextRef.current && inputAudioContextRef.current.state !== 'closed') {
      await inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    if (outputAudioContextRef.current && outputAudioContextRef.current.state !== 'closed') {
        audioPlaybackQueueRef.current.forEach(source => source.stop());
        audioPlaybackQueueRef.current.clear();
        await outputAudioContextRef.current.close();
        outputAudioContextRef.current = null;
    }
  }, []);

  const startConversation = useCallback(async () => {
    if (!process.env.API_KEY) {
      const errorMsg = 'API Key not found. Please make sure the API_KEY environment variable is set.';
      setError(errorMsg);
      setStatus('ERROR');
      console.error(errorMsg);
      return;
    }
    
    setError(null);
    setStatus('CONNECTING');
    setTranscription([]);
    setIsModelThinking(false);
    currentInputTranscriptionRef.current = '';
    currentOutputTranscriptionRef.current = '';

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        microphoneStreamRef.current = stream;

        // FIX: Cast 'window' to 'any' to handle legacy 'webkitAudioContext' for browser compatibility without TypeScript errors.
        inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
        // FIX: Cast 'window' to 'any' to handle legacy 'webkitAudioContext' for browser compatibility without TypeScript errors.
        outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        nextAudioStartTimeRef.current = 0;

        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        
        const langConfig = SUPPORTED_LANGUAGES.find(l => l.name === selectedLanguage);
        const voiceName = langConfig?.voice || 'Zephyr';

        const systemInstruction = `You are a friendly and helpful assistant named Shipra. You were created by CoderAbhi. When asked your name, you must say your name is Shipra. When asked who made you, you must say you were created by CoderAbhi. Be concise in your other responses. The user is speaking ${selectedLanguage}, so you must respond in ${selectedLanguage}.`;

        sessionPromiseRef.current = ai.live.connect({
            model: 'gemini-2.5-flash-native-audio-preview-09-2025',
            config: {
                responseModalities: [Modality.AUDIO],
                speechConfig: {
                    voiceConfig: { prebuiltVoiceConfig: { voiceName: voiceName } },
                },
                inputAudioTranscription: {},
                outputAudioTranscription: {},
                systemInstruction: systemInstruction,
            },
            callbacks: {
                onopen: () => {
                    setStatus('CONNECTED');
                    const source = inputAudioContextRef.current!.createMediaStreamSource(stream);
                    const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
                    scriptProcessorRef.current = scriptProcessor;

                    scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
                        const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
                        const pcmBlob = createPcmBlob(inputData);
                        if (sessionPromiseRef.current) {
                            sessionPromiseRef.current.then((session) => {
                                session.sendRealtimeInput({ media: pcmBlob });
                            });
                        }
                    };
                    source.connect(scriptProcessor);
                    scriptProcessor.connect(inputAudioContextRef.current!.destination);
                },
                onmessage: async (message: LiveServerMessage) => {
                    // Handle transcription
                    if (message.serverContent?.inputTranscription) {
                        currentInputTranscriptionRef.current += message.serverContent.inputTranscription.text;
                        updateTranscription({ speaker: 'user', text: currentInputTranscriptionRef.current });
                    }
                    if (message.serverContent?.outputTranscription) {
                        if (currentOutputTranscriptionRef.current === '') {
                            setIsModelThinking(true);
                        }
                        currentOutputTranscriptionRef.current += message.serverContent.outputTranscription.text;
                        updateTranscription({ speaker: 'model', text: currentOutputTranscriptionRef.current });
                    }
                    if (message.serverContent?.turnComplete) {
                        setTranscription(prev => 
                            prev.map(turn => 
                                turn.isFinal ? turn : { ...turn, isFinal: true }
                            )
                        );
                        currentInputTranscriptionRef.current = '';
                        currentOutputTranscriptionRef.current = '';
                        setIsModelThinking(false);
                    }

                    // Handle audio playback
                    const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                    if (audioData && outputAudioContextRef.current) {
                        setIsModelThinking(false);
                        const outputAudioContext = outputAudioContextRef.current;
                        nextAudioStartTimeRef.current = Math.max(nextAudioStartTimeRef.current, outputAudioContext.currentTime);

                        const audioBuffer = await decodeAudioData(decode(audioData), outputAudioContext, 24000, 1);
                        const source = outputAudioContext.createBufferSource();
                        source.buffer = audioBuffer;
                        source.connect(outputAudioContext.destination);

                        source.addEventListener('ended', () => {
                            audioPlaybackQueueRef.current.delete(source);
                        });

                        source.start(nextAudioStartTimeRef.current);
                        nextAudioStartTimeRef.current += audioBuffer.duration;
                        audioPlaybackQueueRef.current.add(source);
                    }

                    // Handle interruptions
                    if (message.serverContent?.interrupted) {
                        audioPlaybackQueueRef.current.forEach(source => source.stop());
                        audioPlaybackQueueRef.current.clear();
                        nextAudioStartTimeRef.current = 0;
                    }
                },
                onerror: (e: ErrorEvent) => {
                    console.error('Live session error:', e);
                    if (e.message && e.message.toLowerCase().includes('api key')) {
                        setError('Invalid API Key. Please check your key and try again.');
                    } else {
                        setError('A network error occurred. Please check your connection.');
                    }
                    setStatus('ERROR');
                    stopConversation();
                },
                onclose: () => {
                    stopConversation();
                },
            },
        });

        // Handle errors that occur during the initial connection promise
        sessionPromiseRef.current.catch((err: Error) => {
            console.error('Failed to connect to the Live session:', err);
            let errorMessage = 'A network error occurred while connecting.';
            if (err.message) {
                if (err.message.toLowerCase().includes('api key')) {
                    errorMessage = 'Invalid API Key. Please check your key and try again.';
                } else if (err.message.includes('403')) {
                    errorMessage = 'Permission denied. Check your API key and permissions.';
                }
            }
            setError(errorMessage);
            setStatus('ERROR');
            stopConversation();
        });

    } catch (err) {
        console.error(err);
        setError('Failed to initialize microphone. Please grant permission and try again.');
        setStatus('ERROR');
    }
  }, [stopConversation, updateTranscription, selectedLanguage]);
  
  const isConversationActive = status === 'CONNECTING' || status === 'CONNECTED';

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col font-sans">
      <header className="p-4 bg-gray-800/50 backdrop-blur-sm border-b border-gray-700 shadow-lg flex flex-col sm:flex-row sm:justify-between items-center gap-2 sm:gap-4">
        <div>
            <h1 className="text-xl sm:text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400 text-center">
            Shipra Live Conversation
            </h1>
        </div>
        <div>
            <select 
                value={selectedLanguage} 
                onChange={e => setSelectedLanguage(e.target.value as LanguageName)}
                disabled={isConversationActive}
                className="bg-gray-700 border border-gray-600 rounded-md py-1 px-2 text-white focus:ring-purple-500 focus:border-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Select conversation language"
            >
                {SUPPORTED_LANGUAGES.map((lang) => (
                    <option key={lang.name} value={lang.name}>{lang.displayName}</option>
                ))}
            </select>
        </div>
      </header>

      <main className="flex-grow flex flex-col p-4 md:p-6 overflow-hidden">
        <div ref={transcriptionContainerRef} className="flex-grow overflow-y-auto space-y-6 mb-4 pr-2">
          {transcription.map((turn, index) => (
            <div key={index} className={`flex ${turn.speaker === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-xs md:max-w-md lg:max-w-2xl p-3 rounded-2xl ${
                  turn.speaker === 'user' 
                  ? 'bg-blue-600 rounded-br-none' 
                  : 'bg-gray-700 rounded-bl-none'
              } ${!turn.isFinal ? 'opacity-70' : ''}`}>
                <p className="text-sm md:text-base">{turn.text}</p>
              </div>
            </div>
          ))}
          {isModelThinking && (
            <div className="flex justify-start">
              <div className="max-w-xs md:max-w-md lg:max-w-2xl p-3 rounded-2xl bg-gray-700 rounded-bl-none">
                <div className="flex items-center justify-center space-x-1.5 h-5">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.3s]"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse [animation-delay:-0.15s]"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                </div>
              </div>
            </div>
          )}
           {!isConversationActive && transcription.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-gray-500 text-center">
                 <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-16 h-16 mb-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5m-6 7.5a6 6 0 0 1-6-6v-1.5m12 0v-1.5a6 6 0 0 0-6-6v0a6 6 0 0 0-6 6v1.5m12 0v-1.5a6 6 0 0 0-6-6v0a6 6 0 0 0-6 6v1.5" />
                </svg>
              <p className="text-lg">Click the microphone to start a conversation</p>
            </div>
          )}
        </div>
      </main>

      <footer className="sticky bottom-0 left-0 right-0 p-4 bg-gray-900/80 backdrop-blur-sm border-t border-gray-800 flex flex-col items-center justify-center">
        <div className="w-full max-w-md flex flex-col items-center">
            <button
            onClick={isConversationActive ? stopConversation : startConversation}
            className={`w-16 h-16 md:w-20 md:h-20 rounded-full flex items-center justify-center transition-all duration-300 ease-in-out focus:outline-none focus:ring-4 focus:ring-offset-2 focus:ring-offset-gray-900
                ${status === 'CONNECTING' && 'bg-yellow-500 animate-pulse cursor-not-allowed'}
                ${status === 'CONNECTED' && 'bg-red-600 hover:bg-red-700 focus:ring-red-500'}
                ${(status === 'IDLE' || status === 'CLOSED' || status === 'ERROR') && 'bg-green-600 hover:bg-green-700 focus:ring-green-500'}`
            }
            disabled={status === 'CONNECTING'}
            >
                {isConversationActive ? (
                     <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-8 h-8 md:w-10 md:h-10">
                        <path fillRule="evenodd" d="M4.5 7.5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3-3h-9a3 3 0 0 1-3-3v-9Z" clipRule="evenodd" />
                    </svg>
                ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 md:w-10 md:h-10" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
                    </svg>
                )}
            </button>
            <p className="mt-3 text-sm text-gray-400 capitalize min-h-[20px]">
                {error ? <span className="text-red-400">{error}</span> : status}
            </p>
        </div>
      </footer>
    </div>
  );
};

export default App;