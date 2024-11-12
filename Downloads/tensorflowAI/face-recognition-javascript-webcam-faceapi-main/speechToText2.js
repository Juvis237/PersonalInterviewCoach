// npm install assemblyai

import { AssemblyAI } from 'assemblyai'

const client = new AssemblyAI({
  apiKey: "3c1ddeda7933471bb8213fc59a08b48a"
})

const audioUrl =
  'https://assembly.ai/sports_injuries.mp3'

const config = {
  audio_url: audioUrl
}

const run = async () => {
  const transcript = await client.transcripts.transcribe(config)
  console.log(transcript.text)
}

run()