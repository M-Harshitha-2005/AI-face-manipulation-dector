# Day-1
Here is a comprehensive, structured study guide designed to take you from a beginner level to a solid medium level, ensuring you are completely prepared for your next day of classes.

The content is broken down into four clear sections: **Slides L00**, **Slides L01**, the practical terminal data from your **Day-1 Document**, and **Core Peer-to-Peer Linux Concepts** to give you an extra competitive edge.

---

## Section 1: Detailed Breakdown of L00 ("Linux - The Beginning")

This file sets up the history, core traits, and structural architecture of the operating system.

### 1. The Genesis of GNU/Linux

* **What is it?** "Linux" is actually a combination of two things: the **Kernel** (the core engine) and **GNU tools** (the utilities built around it).


* **The Kernel (1991):** Created by **Linus Torvalds**, it is written mostly in C and some Assembly. It handles the lowest-level tasks—communicating directly with your CPU, RAM, and hardware.


* **The GNU Project (1983):** Launched by **Richard Stallman**, this project created the open-source tools, compilers, and software bundles required to make a kernel usable for daily work.



### 2. Five-Layer Architecture

Think of Linux as an onion. You interact with the outside, and it passes instructions inward:

1. **Hardware:** Your physical machine (CPU, RAM, Hard Drives).


2. **Core Kernel & Modules:** The brain that controls the hardware directly.


3. **System Calls (Syscalls):** A bridge of around 400 special entry points (like `open()`, `read()`, `fork()`) that allow applications to request hardware tasks safely.


4. **System Libraries:** Bundles of shared code (like `libc.so`) that applications lean on so programmers don't have to rewrite code from scratch.


5. **Shell & Commands:** The top-layer interface where you type instructions.



### 3. Key Vocabulary & System Architecture Definitions

* **Multi-User / Multi-Tenant:** Multiple users can log in and run processes at the exact same time without breaking each other’s environments.


* **Multi-Tasking / Multi-Processing:** The system can run numerous tasks simultaneously and divide the work efficiently across multiple physical CPU cores.


* **Linux Distribution (Distro):** The raw kernel code combined with a selection of GNU tools, system programs, installer scripts, and an optional visual GUI (like GNOME or KDE). Examples include Ubuntu, Fedora, Debian, and Red Hat (RHEL).


* **Internal vs. External Commands:**
* **Internal (Built-in):** Built directly inside your terminal shell program (e.g., `cd`, `echo`). They run instantly because the shell doesn't have to look for them.


* **External:** Separate executable programs stored in system folders like `/bin` or `/usr/bin` (e.g., `ls`, `mkdir`).





---

## Section 2: Detailed Breakdown of L01 ("Linux Commands")

This file focuses on how to log in, understand the help manuals, control files, track system data, and manage processes.

### 1. Session Management & Discovery Commands

* **`ssh` (Secure Shell):** Used to log into a remote computer securely over an encrypted network connection.


* *Syntax:* `ssh username@ip_address`



* **`uname` (Unix Name):** Prints vital data about the system's core engine.


* `-a` (All): Prints everything.


* `-r` (Release): Shows the active kernel version.


* `-m` (Machine): Shows your hardware architecture (e.g., `x86_64` or `aarch64`).




* **`tty` (Teletypewriter):** Tells you the specific terminal session path you are currently using (e.g., `/dev/pts/1` for a remote connection).


* **`w` / `who` (Who/What):** Shows a clean list of every person logged into the system right now, what terminal they are on, and what command they are executing.


* **`last` (Last Logins):** Scans the system logs backwards to display a chronological list of recent successful login and logout sessions.


* **`exit` / `logout` / `Ctrl+D`:** Safely kills your active terminal shell session and closes your network connection.



### 2. Documentation & System Navigation

* **`man` (Manual):** The built-in, offline instruction manual for nearly every command.


* *Usage:* `man ls` opens the manual pages for the list command.




* **`apropos` (Regarding/About):** A keyword search engine for manuals. If you forget a command name but know what you want to do, type `apropos "keyword"` to find matching tools.



### 3. Basic File Operations

* **`mv` (Move):** Renames a file or shifts it into a different folder. Moving a file within the same filesystem is instant because it only changes the name pointer, not the actual data on disk.


* **`rm` (Remove):** Permanently deletes a file.


* `-i` (Interactive): Asks for confirmation before deleting.


* `-f` (Force): Deletes files without asking questions.


* `-r` (Recursive): Deletes an entire folder and all files/subfolders inside it.


* *Warning:* Linux does **not** have a built-in command-line recycle bin. Running `rm -rf` deletes data permanently.




* **`stat` (Status):** Displays the hidden metadata of a file, including its exact size, modification timestamps, and Inode tracking number.



---

## Section 3: Deep Dive into the "Day-1" Practical Log Document

Your lab log shows how these theoretical concepts apply in a live environment.

### Real-World File Listing (`ls -l /boot`)

Your log demonstrates a detailed listing of the system's boot folder:

```bash
-rw-r--r-- 1 root root   302833 Mar 26 00:01 config-6.17.0-22-generic
drwxr-xr-x 3 root root     4096 Jan  1  1970 efi
lrwxrwxrwx 1 root root       28 Jun  5 06:06 initrd.img -> initrd.img-6.17.0-35-generic

```

Let's break down exactly what every single part of that output means:

#### 1. File Type Indicator (The First Character)

* `-` : A standard **regular file** (like a text file, image, or executable program).


* `dr` : A **directory** (a folder used to store other files).


* `l` : A **symbolic link** (a shortcut or pointer redirecting you to another file).



#### 2. The Permission Notation (`rwxr-xr-x`)

This string is split into three distinct sets of permissions:

* **Set 1 (Characters 2-4):** **Owner** permissions (what the creator can do).


* **Set 2 (Characters 5-7):** **Group** permissions (what members of the file's assigned group can do).


* **Set 3 (Characters 8-10):** **Others** permissions (what everyone else on the system can do).



*The Letters Mean:*

* `r` = **Read** (view content).


* `w` = **Write** (edit or delete content).


* `x` = **Execute** (run the file as a program, or enter the folder).



#### 3. Links, Ownership, and Size Columns

* The number `1` or `3` shows how many hard links points directly to this file on disk.


* The first `root` name tells you the specific **User Owner**.


* The second `root` name tells you the **Group Owner**.


* The number `302833` or `4096` represents the exact size of the file or folder metadata in **Bytes**.


* `Mar 26 00:01` reflects the date and time the file was last modified.


* `config-6.17.0-22-generic` is the actual name of the item on disk.



---

## Section 4: AI Insights — Medium-Level Concepts for Day 2

To give you an advantage in your next class, here are the core mental models and shortcuts used by intermediate Linux administrators.

### 1. The Linux Philosophy: "Everything is a File"

In Windows, hardware configuration is hidden behind a registry. In Linux, almost everything—including hardware—is represented as a plain text file:

* Your hard drive is managed at `/dev/sda`.
* Your active memory status can be read via text at `/proc/meminfo`.
* Your terminal screens are tracked under `/dev/pts/`.

### 2. Relative vs. Absolute Pathways

* **Absolute Paths:** Always start from the absolute root directory (`/`). They work exactly the same way no matter where you are currently standing.
* *Example:* `cd /var/log/nginx/`


* **Relative Paths:** Do not start with a slash. They look for files relative to your current location.
* `.` = Your current directory.
* `..` = The parent directory (one step backward).
* `~` = Your personal user home directory.
* *Example:* `cd ../../bin/` (Moves up two levels, then enters the bin folder).



### 3. Combining Commands (Pipes and Redirection)

On Day 2, you will likely learn how to chain commands together. Here is a head start on how it works:

* **The Pipe (`|`):** Takes the output of one command and feeds it directly as the input to a second command.
* *Example:* `w | grep cse` (Lists all logged-in users, but filters out and shows only lines containing the term "cse").


* **Output Redirection (`>` and `>>`):**
* `>`: Redirects command output away from your screen and writes it into a file (wiping out any text already in that file).
* `>>`: Appends command output to the end of an existing file without deleting what is inside.
* *Example:* `uname -a > system_info.txt`



### 4. Essential Keyboard Shortcuts to Save Time

* **`Tab` Key:** **Auto-completion**. Type the first few letters of a command or file path and hit `Tab` to let the system finish typing it for you. Hitting it twice reveals all matching options.
* **`Ctrl + C`:** **Interrupt signal**. Forcefully stops whatever command or program is currently running or hanging in your terminal.
* **`Ctrl + L`:** Clears your terminal window instantly, giving you a clean slate to type on.
* **`Up / Down Arrow Keys`:** Scroll through your command history so you don't have to retype long strings.

