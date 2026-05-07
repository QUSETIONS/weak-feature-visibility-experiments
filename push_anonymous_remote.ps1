param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteUrl
)

if ($RemoteUrl -notmatch '^https://github\.com/[^/]+/[^/]+(\.git)?$') {
    Write-Error "RemoteUrl must look like https://github.com/<anonymous-account>/<repo>.git"
    exit 1
}

$active = gh auth status 2>&1 | Out-String
if ($active -match 'Logged in to github\.com account') {
    Write-Warning "This machine may have an identifiable GitHub account configured in gh. Do not use gh repo create or gh auth with a non-anonymous account for this repository."
}

git remote remove origin 2>$null
git remote add origin $RemoteUrl
git branch -M main
git push -u origin main
