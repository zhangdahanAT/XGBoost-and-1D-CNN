<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Results of UTR variants</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            margin: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }

        h1 {
            text-align: center;
            font-size: 28px;
            margin-bottom: 20px;
        }

        h2 {
            margin-top: 40px;
            text-align: center;
        }

        .heatmap-container {
            display: flex;
            justify-content: center;
            width: 100%;
            max-width: 1080px;
            height: 1080px;
            margin-bottom: 20px;
        }

        .heatmap {
            display: grid;
            grid-template-columns: repeat(6, 180px);
            grid-template-rows: repeat(6, 180px);
            gap: 4px;
            justify-content: center;
        }

        .heatmap-cell {
            width: 180px;
            height: 180px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            border-radius: 4px;
            border: 1px solid #999;
            cursor: pointer;
            text-shadow: 0 0 2px #000;
            font-size: 24px;
        }

        .colorbar-container {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
            margin-top: 40px;
        }

        .colorbar {
            height: 20px;
            width: 300px;
            background: linear-gradient(to right, 
                rgb(13,8,135), 
                rgb(84,3,160), 
                rgb(139,10,165), 
                rgb(186,54,130), 
                rgb(222,119,93), 
                rgb(249,180,51), 
                rgb(240,249,33));
            border: 1px solid #333;
            margin: 0 10px;
        }

        .colorbar-labels {
            display: flex;
            justify-content: space-between;
            width: 300px;
            font-size: 14px;
            margin: 0 auto;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 40px;
        }

        th, td {
            border: 1px solid #999;
            padding: 8px;
            text-align: center;
        }

        th {
            background-color: #f3f3f3;
        }

        .prediction-warning {
            width: 100%;
            max-width: 1080px;
            box-sizing: border-box;
            margin: 10px 0 20px;
            padding: 12px;
            border: 1px solid #d8a200;
            border-radius: 4px;
            background: #fff8d8;
            color: #6d5200;
            text-align: center;
        }
    </style>
</head>
<body>

<h1>Results of UTR variants</h1>

<h2>Expected Translation Efficiency</h2>

<?php
$gc_thresholds = [0, 0.49029314, 0.56321839, 0.61463415, 0.65753722, 0.70317742, 1];
$bracket_thresholds = [0, 0.32142857, 0.43410853, 0.5034965, 0.55746662, 0.6122449, 0.77966102];
$heatmap_counts = array_fill(0, 6, array_fill(0, 6, 0));
$heatmap_te_sums = array_fill(0, 6, array_fill(0, 6, 0.0));
$heatmap_te_counts = array_fill(0, 6, array_fill(0, 6, 0));

/*
 * The Python executable must belong to the environment in which xgboost,
 * numpy, pandas, and the other model dependencies are installed. For a Conda
 * deployment, set UTR_TE_PYTHON in the PHP-FPM environment or replace the
 * fallback path below with the output of: which python
 */
$python_binary = getenv('UTR_TE_PYTHON') ?: '/usr/bin/python3';
$model_directory = '/www/wwwroot/utrtransfomer_cn/utrwebsite/static/model';
$species_name = 'rice';
$prediction_error = null;

/**
 * Predict all rows in one Python process. This avoids loading XGBoost once per
 * sequence. Every command argument is escaped before it reaches proc_open.
 */
function predict_translation_efficiency_batch($inputs, $species, $python_binary, $model_directory, &$error_message)
{
    $error_message = null;
    if (empty($inputs)) {
        return [];
    }
    if (!is_executable($python_binary)) {
        $error_message = 'Configured Python executable is unavailable.';
        return [];
    }
    if (!is_dir($model_directory)) {
        $error_message = 'Configured model directory is unavailable.';
        return [];
    }
    if (!function_exists('proc_open')) {
        $error_message = 'The PHP proc_open function is disabled.';
        return [];
    }

    $input_file = tempnam(sys_get_temp_dir(), 'utr_te_');
    if ($input_file === false) {
        $error_message = 'Unable to create a temporary prediction input.';
        return [];
    }

    $payload = [
        'species' => $species,
        'rows' => array_values($inputs),
    ];
    $written = file_put_contents(
        $input_file,
        json_encode($payload, JSON_UNESCAPED_SLASHES)
    );
    if ($written === false) {
        @unlink($input_file);
        $error_message = 'Unable to write the temporary prediction input.';
        return [];
    }

    $python_code = <<<'PYTHON'
import json
import sys

model_dir, input_file = sys.argv[1], sys.argv[2]
sys.path.insert(0, model_dir)
from predict_te_from_species import predict_te

with open(input_file, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

results = []
for row in payload["rows"]:
    try:
        prediction = predict_te(
            payload["species"],
            row["gc_content"],
            row["stem_ratio"],
            model_dir,
        )
        results.append({
            "ok": True,
            "predicted_TE": prediction["predicted_TE"],
            "warnings": prediction.get("warnings", []),
        })
    except Exception as exc:
        results.append({"ok": False, "error": str(exc)})

print(json.dumps({"results": results}, ensure_ascii=False))
PYTHON;

    $command = escapeshellarg($python_binary)
        . ' -c ' . escapeshellarg($python_code)
        . ' ' . escapeshellarg($model_directory)
        . ' ' . escapeshellarg($input_file);
    $descriptor_spec = [
        0 => ['pipe', 'r'],
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];
    $pipes = [];
    $process = @proc_open($command, $descriptor_spec, $pipes);
    if (!is_resource($process)) {
        @unlink($input_file);
        $error_message = 'Unable to start the TE prediction process.';
        return [];
    }

    fclose($pipes[0]);
    $stdout = stream_get_contents($pipes[1]);
    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    $exit_code = proc_close($process);
    @unlink($input_file);

    if ($exit_code !== 0) {
        $error_message = 'TE prediction process failed: ' . trim($stderr);
        return [];
    }

    $stdout_lines = preg_split('/\R/', trim($stdout));
    $json_line = !empty($stdout_lines) ? end($stdout_lines) : '';
    $decoded = json_decode($json_line, true);
    if (!is_array($decoded) || !isset($decoded['results']) || !is_array($decoded['results'])) {
        $error_message = 'TE prediction returned an invalid response.';
        if (trim($stderr) !== '') {
            $error_message .= ' ' . trim($stderr);
        }
        return [];
    }
    return $decoded['results'];
}

$csv_path = __DIR__ . '/rna_results.csv';
$all_sequences = [];
$prediction_inputs = [];
$wt_position = null;
$header = [];

$file = @fopen($csv_path, 'r');
if ($file === false) {
    $prediction_error = 'Unable to open rna_results.csv.';
} else {
    $header = fgetcsv($file);
    if (!is_array($header)) {
        $header = [];
        $prediction_error = 'rna_results.csv has no valid header.';
    } elseif (!in_array('GC Content (%)', $header, true) ||
              !in_array('Base-pairing potential (%)', $header, true)) {
        $prediction_error = 'Required GC content or base-pairing potential column is missing.';
    } else {
        while (($data = fgetcsv($file)) !== false) {
            if (count($data) !== count($header)) {
                continue;
            }
            $row = array_combine($header, $data);
            $gc = floatval($row['GC Content (%)']) / 100.0;
            $br = floatval($row['Base-pairing potential (%)']) / 100.0;
            $row['GC'] = $gc;
            $row['BR'] = $br;
            $row['Predicted TE'] = null;
            $row['Prediction warning'] = '';
            $all_sequences[] = $row;
            $prediction_inputs[] = [
                'gc_content' => $gc,
                'stem_ratio' => $br,
            ];
        }
    }
    fclose($file);
}

if ($prediction_error === null) {
    $prediction_results = predict_translation_efficiency_batch(
        $prediction_inputs,
        $species_name,
        $python_binary,
        $model_directory,
        $prediction_error
    );
    if ($prediction_error === null && count($prediction_results) !== count($all_sequences)) {
        $prediction_error = 'The number of TE predictions did not match the number of CSV rows.';
    }
    if ($prediction_error === null) {
        foreach ($prediction_results as $index => $prediction_result) {
            if (!empty($prediction_result['ok']) &&
                isset($prediction_result['predicted_TE']) &&
                is_numeric($prediction_result['predicted_TE'])) {
                $all_sequences[$index]['Predicted TE'] = floatval($prediction_result['predicted_TE']);
                $warnings = isset($prediction_result['warnings']) && is_array($prediction_result['warnings'])
                    ? $prediction_result['warnings'] : [];
                $all_sequences[$index]['Prediction warning'] = implode('; ', $warnings);
            } else {
                $all_sequences[$index]['Prediction warning'] = 'Prediction failed';
                if (isset($prediction_result['error'])) {
                    error_log('[5UTRDesigner TE row ' . $index . '] ' . $prediction_result['error']);
                }
            }
        }
    }
}

foreach ($all_sequences as $index => $row) {
    $gc = $row['GC'];
    $br = $row['BR'];
    $gc_index = 5;
    $br_index = 5;
    for ($i = 0; $i < 6; $i++) {
        if ($gc >= $gc_thresholds[$i] && $gc < $gc_thresholds[$i + 1]) {
            $gc_index = $i;
            break;
        }
    }
    for ($j = 0; $j < 6; $j++) {
        if ($br >= $bracket_thresholds[$j] && $br < $bracket_thresholds[$j + 1]) {
            $br_index = $j;
            break;
        }
    }
    $heatmap_counts[$br_index][$gc_index]++;
    if (is_numeric($row['Predicted TE'])) {
        $heatmap_te_sums[$br_index][$gc_index] += floatval($row['Predicted TE']);
        $heatmap_te_counts[$br_index][$gc_index]++;
    }
    $all_sequences[$index]['Heatmap row'] = $br_index;
    $all_sequences[$index]['Heatmap column'] = $gc_index;
    if (isset($row['Sequence ID']) && $row['Sequence ID'] === 'WT') {
        $wt_position = [$br_index, $gc_index];
    }
}

$heatmap_te = array_fill(0, 6, array_fill(0, 6, null));
$all_predicted_te = [];
for ($i = 0; $i < 6; $i++) {
    for ($j = 0; $j < 6; $j++) {
        if ($heatmap_te_counts[$i][$j] > 0) {
            $heatmap_te[$i][$j] = $heatmap_te_sums[$i][$j] / $heatmap_te_counts[$i][$j];
            $all_predicted_te[] = $heatmap_te[$i][$j];
        }
    }
}
$te_color_min = count($all_predicted_te) ? min($all_predicted_te) : 0.0;
$te_color_max = count($all_predicted_te) ? max($all_predicted_te) : 1.0;
if ($te_color_min == $te_color_max) {
    $te_color_max = $te_color_min + 1.0;
}

$display_header = $header;
$display_header[] = 'Predicted TE';
$display_header[] = 'Prediction warning';

if ($prediction_error !== null) {
    error_log('[5UTRDesigner TE] ' . $prediction_error);
    echo '<div class="prediction-warning">TE prediction is temporarily unavailable. Please check the server prediction environment.</div>';
}

echo "<script>\n";
echo "const frequencyData = " . json_encode($heatmap_counts) . ";\n";
echo "const predictedTeData = " . json_encode($heatmap_te) . ";\n";
echo "const predictedTeMin = " . json_encode($te_color_min) . ";\n";
echo "const predictedTeMax = " . json_encode($te_color_max) . ";\n";
echo "const allSequences = " . json_encode($all_sequences) . ";\n";
echo "const csvHeader = " . json_encode($display_header) . ";\n";
echo "const wildTypeCell = " . json_encode($wt_position) . ";\n";
echo "</script>\n";
?>

<div class="heatmap-container">
    <div class="heatmap" id="heatmap"></div>
</div>

<div class="colorbar-container">
    <span id="colorbar-min"></span>
    <div class="colorbar"></div>
    <span id="colorbar-max"></span>
</div>

<div class="colorbar-labels">
    <span>Low</span>
    <span>High</span>
</div>

<h2>RNA Sequence Analysis Table</h2>

<table>
    <thead>
    <tr>
        <th>Sequence ID</th>
        <th>Variant sequence</th>
        <th>Structure</th>
        <th>GC Content (%)</th>
        <th>Base-pairing potential (%)</th>
        <th>Editing strategy</th>
        <th>Free energy (kcal/mol)</th>
        <th>Predicted TE</th>
        <th>Prediction warning</th>
    </tr>
    </thead>
    <tbody>
    <?php
    foreach ($all_sequences as $row) {
        echo "<tr>";
        foreach ($display_header as $column_name) {
            $cell = isset($row[$column_name]) ? $row[$column_name] : '';
            if ($column_name === 'Predicted TE' && is_numeric($cell)) {
                $cell = number_format(floatval($cell), 6, '.', '');
            }
            echo "<td>" . htmlspecialchars((string)$cell, ENT_QUOTES, 'UTF-8') . "</td>";
        }
        echo "</tr>";
    }
    ?>
    </tbody>
</table>

<script>
function plasmaColor(value) {
    const denominator = predictedTeMax - predictedTeMin;
    const t = denominator > 0
        ? Math.min(1, Math.max(0, (value - predictedTeMin) / denominator))
        : 0.5;
    const colors = [
        [13, 8, 135], [84, 3, 160], [139, 10, 165],
        [186, 54, 130], [222, 119, 93], [249, 180, 51], [240, 249, 33]
    ];
    const scaledIndex = t * (colors.length - 1);
    const idx = Math.floor(scaledIndex);
    const frac = scaledIndex - idx;
    const c1 = colors[idx];
    const c2 = colors[Math.min(idx + 1, colors.length - 1)];
    const r = Math.round(c1[0] + frac * (c2[0] - c1[0]));
    const g = Math.round(c1[1] + frac * (c2[1] - c1[1]));
    const b = Math.round(c1[2] + frac * (c2[2] - c1[2]));
    return `rgb(${r}, ${g}, ${b})`;
}

const heatmap = document.getElementById("heatmap");
document.getElementById("colorbar-min").textContent = predictedTeMin.toFixed(3);
document.getElementById("colorbar-max").textContent = predictedTeMax.toFixed(3);

predictedTeData.forEach((row, rowIndex) => {
    row.forEach((val, colIndex) => {
        const cell = document.createElement("div");
        cell.className = "heatmap-cell";
        const hasPrediction = typeof val === "number" && Number.isFinite(val);
        cell.style.backgroundColor = hasPrediction ? plasmaColor(val) : "#bdbdbd";
        const freq = frequencyData[rowIndex][colIndex];

        let wildTypeLabel = "";
        if (wildTypeCell && rowIndex === wildTypeCell[0] && colIndex === wildTypeCell[1]) {
            wildTypeLabel = `<div style="color: green; font-weight: bold; font-size: 18px;">Wild type</div>`;
        }

        cell.innerHTML = `
            <div>${hasPrediction ? val.toFixed(3) : "N/A"}</div>
            <div style="font-size:24px;">n=${freq}</div>
            ${wildTypeLabel}
        `;

        cell.title = `(${rowIndex + 1}, ${colIndex + 1}) mean predicted TE: ${hasPrediction ? val.toFixed(6) : "N/A"}\nFrequency: ${freq}`;
        cell.onclick = () => {
            const matched = allSequences.filter(seq =>
                seq["Heatmap row"] === rowIndex &&
                seq["Heatmap column"] === colIndex
            );

            if (matched.length === 0) {
                alert(`No sequences found in Cell (${rowIndex + 1}, ${colIndex + 1})`);
                return;
            }

            const rows = [csvHeader.join(",")];
            matched.forEach(seq => {
                const row = csvHeader.map(key => `"${String(seq[key] ?? "").replace(/"/g, '""')}"`);
                rows.push(row.join(","));
            });

            const blob = new Blob([rows.join("\n")], { type: "text/csv" });
            const filename = `expected_TE_row${rowIndex + 1}_col${colIndex + 1}.csv`;
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };

        heatmap.appendChild(cell);
    });
});
</script>

</body>
</html>
