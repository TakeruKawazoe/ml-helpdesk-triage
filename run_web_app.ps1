$ErrorActionPreference = "Stop"

function Get-ConfiguredValue {
    param([Parameter(Mandatory = $true)][string]$Name)

    $processValue = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not [string]::IsNullOrWhiteSpace($processValue)) {
        return $processValue
    }
    return [Environment]::GetEnvironmentVariable($Name, "User")
}

$notionApiToken = Get-ConfiguredValue -Name "NOTION_API_TOKEN"
$notionDatabaseId = Get-ConfiguredValue -Name "NOTION_DATABASE_ID"

if ([string]::IsNullOrWhiteSpace($notionApiToken)) {
    throw "NOTION_API_TOKEN is not set in the user environment."
}

if ([string]::IsNullOrWhiteSpace($notionDatabaseId)) {
    throw "NOTION_DATABASE_ID is not set in the user environment."
}

$env:NOTION_API_TOKEN = $notionApiToken
$env:NOTION_DATABASE_ID = $notionDatabaseId

$slackVariableNames = @(
    "SLACK_BOT_TOKEN",
    "SLACK_CHANNEL_ID",
    "SLACK_MENTION_SOMU",
    "SLACK_MENTION_KEIRI",
    "SLACK_MENTION_JOSYS",
    "SLACK_MENTION_DEVELOPMENT",
    "SLACK_MENTION_INFRASTRUCTURE"
)

foreach ($variableName in $slackVariableNames) {
    $value = Get-ConfiguredValue -Name $variableName
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        Set-Item -Path "Env:$variableName" -Value $value
    }
}

if (-not (Test-Path -LiteralPath "models\improved\manifest.json")) {
    throw "Improved model artifacts are missing. Run: python src\train_improved.py"
}

python src\web_app.py

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
