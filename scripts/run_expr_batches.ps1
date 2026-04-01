param(
    [int]$BatchSize = 24,
    [UInt64]$SearchStart = 136279841,
    [int]$MaxPrimeChecks = 1000000,
    [int]$KnownLimit = 0
)

$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot
$exe = Join-Path (Split-Path $PSScriptRoot -Parent) 'bin' 'll_fft.exe'
$src = Join-Path (Split-Path $PSScriptRoot -Parent) 'src' 'll_fft.cpp'

if (-not (Test-Path $exe)) {
    Write-Host "[$(Get-Date)] Building ll_fft.exe..."
    & g++ -O3 -std=c++17 -march=native -mtune=native -Wall -Wextra -Wpedantic -o $exe $src
    if ($LASTEXITCODE -ne 0) {
        throw 'Build failed.'
    }
}

$knownExponents = @(
    2,3,5,7,13,17,19,31,61,89,107,127,
    521,607,1279,2203,2281,3217,4253,4423,9689,9941,11213,19937,
    21701,23209,44497,86243,110503,132049,216091,756839,859433,1257787,1398269,
    2976221,3021377,6972593,13466917,20996011,24036583,25964951,30402457,
    32582657,37156667,42643801,43112609,57885161,74207281,77232917,82589933,
    136279841
)

if ($KnownLimit -gt 0) {
    $take = [Math]::Min($KnownLimit, $knownExponents.Count)
    $knownExponents = $knownExponents[0..($take - 1)]
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$csvPath = Join-Path $PSScriptRoot ("expr_batch_results_{0}.csv" -f $stamp)
$rows = New-Object System.Collections.Generic.List[object]

function Invoke-ExprRun {
    param([UInt64]$Exponent)

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $output = & $script:exe $Exponent 2>&1
    $exitCode = $LASTEXITCODE
    $sw.Stop()

    if ($exitCode -ne 0) {
        $msg = (($output | Out-String).Trim() -replace '[\r\n]+',' ')
        return [pscustomobject]@{
            Exponent = $Exponent
            Result = "ERROR:$msg"
            ElapsedMs = [int64]$sw.ElapsedMilliseconds
            ExitCode = $exitCode
        }
    }

    $resultLine = ($output | Select-Object -Last 1).ToString().Trim()
    return [pscustomobject]@{
        Exponent = $Exponent
        Result = $resultLine
        ElapsedMs = [int64]$sw.ElapsedMilliseconds
        ExitCode = 0
    }
}

function Test-Prime64 {
    param([UInt64]$n)

    if ($n -lt 2) { return $false }
    if ($n -eq 2 -or $n -eq 3) { return $true }
    if (($n % 2) -eq 0) { return $false }

    [UInt64]$d = $n - 1
    [int]$s = 0
    while (($d % 2) -eq 0) {
        $d /= 2
        $s++
    }

    $bases = @(2,3,5,7,11,13,17)
    foreach ($a in $bases) {
        if ([UInt64]$a -ge $n) { continue }

        $x = [System.Numerics.BigInteger]::ModPow(
            [System.Numerics.BigInteger]$a,
            [System.Numerics.BigInteger]$d,
            [System.Numerics.BigInteger]$n)

        if ($x -eq 1 -or $x -eq ($n - 1)) { continue }

        $witnessPassed = $false
        for ($r = 1; $r -lt $s; $r++) {
            $x = [System.Numerics.BigInteger]::Remainder(
                $x * $x,
                [System.Numerics.BigInteger]$n)
            if ($x -eq ($n - 1)) {
                $witnessPassed = $true
                break
            }
        }

        if (-not $witnessPassed) { return $false }
    }

    return $true
}

Write-Host "=== Running known exponents in batches of $BatchSize ==="

$idx = 0
$batch = 0
for ($start = 0; $start -lt $knownExponents.Count; $start += $BatchSize) {
    $batch++
    $end = [Math]::Min($start + $BatchSize - 1, $knownExponents.Count - 1)
    Write-Host ""
    Write-Host ("[Batch {0}] indices {1}..{2}" -f $batch, ($start + 1), ($end + 1))

    for ($i = $start; $i -le $end; $i++) {
        $idx++
        [UInt64]$exp = $knownExponents[$i]
        $run = Invoke-ExprRun -Exponent $exp

        $rows.Add([pscustomobject]@{
            kind = 'KNOWN'
            index = $idx
            batch = $batch
            exp = $run.Exponent
            elapsed_ms = $run.ElapsedMs
        })

        Write-Host ("KNOWN,{0},{1},{2},{3}" -f $idx, $batch, $run.Exponent, $run.ElapsedMs)
    }
}

Write-Host ""
Write-Host "=== Searching first prime exponent >= $SearchStart with result 0 ==="

[UInt64]$candidate = $SearchStart
if ($candidate -le 2) {
    $candidate = 2
} elseif (($candidate % 2) -eq 0) {
    $candidate++
}

$primeChecks = 0
$found = $false

while ($primeChecks -lt $MaxPrimeChecks) {
    if (Test-Prime64 -n $candidate) {
        $primeChecks++
        $run = Invoke-ExprRun -Exponent $candidate

        $rows.Add([pscustomobject]@{
            kind = 'SEARCH'
            index = $primeChecks
            batch = '-'
            exp = $run.Exponent
            result = $run.Result
            elapsed_ms = $run.ElapsedMs
        })

        Write-Host ("SEARCH,{0},-,{1},{2},{3}" -f $primeChecks, $run.Exponent, $run.Result, $run.ElapsedMs)

        if ($run.Result -eq 'Residuo = 0') {
            $rows.Add([pscustomobject]@{
                kind = 'FOUND'
                index = ''
                batch = ''
                exp = $run.Exponent
                result = '0'
                elapsed_ms = ''
            })

            Write-Host ""
            Write-Host "First prime exponent with result 0: $candidate"
            $found = $true
            break
        }
    }

    if ($candidate -eq 2) {
        $candidate = 3
    } else {
        $candidate += 2
    }
}

if (-not $found) {
    Write-Host ""
    Write-Host "Search stopped after $primeChecks prime checks (MaxPrimeChecks=$MaxPrimeChecks)."
}

$rows | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host ""
Write-Host "Finished. Results saved in $csvPath"
