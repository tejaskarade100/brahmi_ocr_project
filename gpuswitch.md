# Free-Tier GPU Relay Guide (Colab & Kaggle)

Building a "Free Tier GPU Relay" takes just a tiny bit of organization, but once you figure it out, it unlocks virtually unlimited free compute!

Here is your master instruction manual for seamlessly switching and resuming TrOCR training across multiple Google Colab and Kaggle accounts.

***

### 🧱 The 2 Required Files
No matter which platform you switch to, you strictly need **only two things**:
1. **`dataset.zip`** (Constant, never changes)
2. **`checkpoint-XYZ` folder** (The latest brain weights saved from the *previous* run)

---

### 🔄 Case 1: Colab (Account A) -> Colab (Account B)
*This is the easiest switch.*
1. On **Account A**, you ran `colab_training.ipynb`. It successfully saved the latest `checkpoint-XYZ` folder into `Account A's` Google Drive (`/MyDrive/Project/brahmi_ocr_project/model/brahmi_trocr/checkpoint-XYZ`).
2. Log into **Account B**. 
3. **The Share Trick:** In Google Drive on Account A, right click the `brahmi_trocr` folder and share it with Account B.
4. On **Account B**, go to "Shared with me", right-click that folder, and click **"Add shortcut to Drive"**. Place the shortcut at the exact path: `MyDrive/Project/brahmi_ocr_project/model/brahmi_trocr/`.
5. Run the `colab_training.ipynb` on Account B. It will see the shortcut, load the weights from Account A's checkpoint, and instantly resume!

---

### 🔄 Case 2: Colab -> Kaggle (Account A)
*When Colab runs out of time and you want to use Kaggle for the rest of the week.*
1. Go to your Google Drive and **Download** the latest `checkpoint-XYZ` folder to your actual computer. Zip it up as `checkpoint.zip`.
2. Open your brand new Kaggle Notebook (`kaggle_training.ipynb`).
3. Click **Add Data -> Upload Data (New Dataset)**. 
4. Upload `dataset.zip` and `checkpoint.zip`.
5. Kaggle will permanently mount them as read-only drives. For example: `/kaggle/input/brahmi-checkpoint/checkpoint-XYZ`.
6. **Critical Step:** You cannot train on a read-only drive! Before running the training cell in Kaggle, copy the checkpoint into the working directory:
   `!cp -r /kaggle/input/brahmi-checkpoint/checkpoint-XYZ /kaggle/working/checkpoints/`
7. Run the training cell! It will automatically detect `/kaggle/working/checkpoints/checkpoint-XYZ` and resume.

---

### 🔄 Case 3: Kaggle (Account A) -> Kaggle (Account B)
*The most powerful and seamless trick of all.*
1. On Kaggle Account A, your training finished. You clicked **"Save Version" -> "Save & Run All (Commit)"**.
2. When the commit finishes, go to the notebook's "Output" tab. You will see your `checkpoints` folder sitting there safely.
3. Log into **Kaggle Account B**. Create a new notebook.
4. Click **Add Data -> Notebook Output Files**.
5. Search for the name of the notebook from Account A and add it!
6. Account A's output is now instantly mounted to Account B! (`/kaggle/input/account-a-notebook-name/checkpoints/checkpoint-XYZ`)
7. Just like Case 2, copy it to the working directory: 
`!cp -r /kaggle/input/.../checkpoints/checkpoint-XYZ /kaggle/working/checkpoints/`
8. Resume training!

---

### 🔄 Case 4: Kaggle -> Back to Google Colab
1. Go to your finished Kaggle notebook's "Output" tab.
2. Click the tiny **Download button** next to the `checkpoint-XYZ` folder.
3. Upload that folder straight into your Google Drive path: `MyDrive/Project/brahmi_ocr_project/model/brahmi_trocr/checkpoint-XYZ`.
4. Open Colab and run `colab_training.ipynb`. It will detect the weights from Drive and resume perfectly.

---

### 🧠 The Golden Rule of Resuming
As long as the `output_dir` in your `train.py` argument contains a folder named `checkpoint-[number]`, HuggingFace `Trainer` will **always** ignore everything else and auto-resume from that exact number. It doesn't care if the checkpoint came from Colab, Kaggle, or a friend's computer!
