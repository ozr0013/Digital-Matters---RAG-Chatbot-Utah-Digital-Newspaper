$ProjectDir = "C:\Users\u1529771\Desktop\ORAG\Digital-Matters---RAG-Chatbot-Utah-Digital-Newspaper"
$BatchFile  = "$ProjectDir\launch_chatbot.bat"
$Desktop    = [Environment]::GetFolderPath('Desktop')
$LinkPath   = "$Desktop\Utah Newspapers RAG Chatbot.lnk"

$Shell    = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($LinkPath)

$Shortcut.TargetPath       = $BatchFile
$Shortcut.WorkingDirectory = $ProjectDir
$Shortcut.Description      = "Utah Digital Newspapers RAG Chatbot"
$Shortcut.WindowStyle      = 1  # Normal window

$Shortcut.Save()

Write-Host "Desktop shortcut created at: $LinkPath"
