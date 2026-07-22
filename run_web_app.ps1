$ErrorActionPreference = "Stop"

$notionApiToken = [Environment]::GetEnvironmentVariable(
    "NOTION_API_TOKEN",
    "User"
)
$notionDatabaseId = [Environment]::GetEnvironmentVariable(
    "NOTION_DATABASE_ID",
    "User"
)

if ([string]::IsNullOrWhiteSpace($notionApiToken)) {
    throw "NOTION_API_TOKEN is not set in the user environment."
}

if ([string]::IsNullOrWhiteSpace($notionDatabaseId)) {
    throw "NOTION_DATABASE_ID is not set in the user environment."
}

$env:NOTION_API_TOKEN = $notionApiToken
$env:NOTION_DATABASE_ID = $notionDatabaseId

python src\web_app.py

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
