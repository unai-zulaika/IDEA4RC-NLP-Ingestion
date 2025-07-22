import React, { useEffect, useState, useRef, useMemo } from "react";
import Box from "@mui/joy/Box";
import Stack from "@mui/joy/Stack";
import Typography from "@mui/joy/Typography";
import { DataGrid } from "@mui/x-data-grid";
import type { GridColDef } from "@mui/x-data-grid/models";
import Tabs from "@mui/joy/Tabs";
import TabList from "@mui/joy/TabList";
import Tab, { tabClasses } from "@mui/joy/Tab";
import TabPanel from "@mui/joy/TabPanel";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import RemoveCircleOutlineIcon from "@mui/icons-material/RemoveCircleOutline";
import WarningAmberIcon from "@mui/icons-material/WarningAmber";
import { saveAs } from "file-saver";
import PatientTimeline from "../components/PatientsTimeline";

// Utility to fetch JSON from public/data folder
async function loadJson<T = any>(fileName: string): Promise<T> {
    const res = await fetch(`/data/results/${fileName}`);
    if (!res.ok) throw new Error(`Failed to fetch ${fileName}: ${res.statusText}`);
    return res.json();
}

export default function Results() {
    // Data states
    const [summaryData, setSummaryData] = useState<any[]>([]);
    const [patientDataRows, setPatientDataRows] = useState<any[]>([]);
    const [qcDataRows, setQcDataRows] = useState<any[]>([]);
    const [variableDataRows, setVariableDataRows] = useState<any[]>([]);
    const [datasourceDataRows, setDatasourceDataRows] = useState<any[]>([]);
    const [importanceDataRows, setImportanceDataRows] = useState<any[]>([]);
    const [phaseDataRows, setPhaseDataRows] = useState<any[]>([]);
    const [patientImpRows, setPatientImpRows] = useState<any[]>([]);
    const [phasePatientRows, setPhasePatientRows] = useState<any[]>([]);
    const [phaseRows, setPhaseRows] = useState<any[]>([]);

    // Modal & tabs
    const [timelineOpen, setTimelineOpen] = useState(false);
    const [timelineRows, setTimelineRows] = useState<any[]>([]);
    const [timelinePID, setTimelinePID] = useState<string>("");
    const [failModalOpen, setFailModalOpen] = useState(false);
    const [failModalRows, setFailModalRows] = useState<any[]>([]);
    const [failModalPatient, setFailModalPatient] = useState<string>("");
    const [activeTab, setActiveTab] = useState<number>(0);

    // Refs for lookups
    const patientFailures = useRef<Map<string, any[]>>(new Map());
    const patientTimeline = useRef<Record<string, any[]>>({});

    // List of result files to load
    const resultFiles = [
        "results.json",
        // add additional result files here
    ];

    // Ensure both percentages exist
    const withTwoPercents = (row: any) => {
        if (row.MissingPercent == null && row.PassPercent != null) {
            const p = parseFloat(String(row.PassPercent).replace("%", ""));
            row.MissingPercent = (100 - p).toFixed(1) + "%";
        } else if (row.PassPercent == null && row.MissingPercent != null) {
            const m = parseFloat(String(row.MissingPercent).replace("%", ""));
            row.PassPercent = (100 - m).toFixed(1) + "%";
        }
        return row;
    };

    // Build patientFailures map
    useEffect(() => {
        (async () => {
            const map = new Map<string, any[]>();
            const seen = new Set<string>();
            for (const fname of resultFiles) {
                try {
                    const doc: any = await loadJson(fname);
                    const runs = Object.values(doc.run_results ?? {});
                    for (const run of runs) {
                        const entityName = fname === "results.json"
                            ? "non_repeatable"
                            : fname.replace(/^results_|\.json$/g, "");
                        ((run as any).validation_result?.results ?? []).forEach((res: any) => {
                            if (res.success) return;
                            const varName = res.expectation_config?.kwargs?.column ?? "Table-level";
                            const qcName = res.expectation_config?.expectation_type ?? "N/A";
                            (res.result?.partial_unexpected_index_list ?? []).forEach((idxObj: any) => {
                                const pid = String(idxObj.patient_id ?? idxObj.PatientID ?? "UNKNOWN");
                                const bad = String(idxObj[varName] ?? idxObj.value ?? "NULL");
                                const key = `${pid}|${varName}|${qcName}|${bad}|${entityName}`;
                                if (seen.has(key)) return;
                                seen.add(key);
                                const row = { id: key, PatientID: pid, Variable: varName, QC: qcName, BadValue: bad, Entity: entityName };
                                if (!map.has(pid)) map.set(pid, []);
                                map.get(pid)!.push(row);
                            });
                        });
                    }
                } catch (e) {
                    console.error(`Error loading ${fname}:`, e);
                }
            }
            patientFailures.current = map;
        })();
    }, []);

    // Summary data
    useEffect(() => {
        (async () => {
            try {

                const dimensionSummary: any = await loadJson("dimension_summary_results.json");
                const order = ["Plausibility", "Conformance", "Completeness", "Total"];
                let id = 1;
                const rows = order.map(cat => withTwoPercents({
                    id: id++,
                    category: cat,
                    Pass: dimensionSummary[cat]?.Passed || 0,
                    Fail: dimensionSummary[cat]?.Failed || 0,
                    Amount: dimensionSummary[cat]?.Total || 0,
                    PassPercent: dimensionSummary[cat]?.PercentagePass ?? "0.0%",
                }));
                setSummaryData(rows);
            } catch (e) {
                console.error("Error loading dimension_summary_results.json:", e);
                setSummaryData([]);
            }
        })();
    }, []);

    // Patient summary
    useEffect(() => {
        (async () => {
            try {
                const data: any[] = await loadJson("patient_summary_results.json");
                if (!Array.isArray(data)) throw new Error("Not an array");
                const rows = data.map((item, i) => withTwoPercents({
                    id: item.PatientID || i,
                    PatientID: item.PatientID,
                    isPassed: item.Failed === 0,
                    Pass: item["Number of Passed Tests"],
                    Fail: item.Failed,
                    Amount: item.Total,
                    PassPercent: typeof item.PercentagePass === 'number'
                        ? item.PercentagePass.toFixed(2) + "%"
                        : item.PercentagePass,
                }));
                setPatientDataRows(rows);
            } catch (e) {
                console.error("Error loading patient_summary_results.json:", e);
                setPatientDataRows([]);
            }
        })();
    }, []);

    // QC summary
    useEffect(() => {
        (async () => {
            try {
                const data: any[] = await loadJson("qc_summary_results.json");
                if (!Array.isArray(data)) throw new Error("Not an array");
                const rows = data.map((item, i) => withTwoPercents({
                    id: item.ge_name || i,
                    QC: item.ge_name,
                    isPassed: item.failed_checks === 0,
                    Pass: item.passed_checks,
                    Fail: item.failed_checks,
                    Amount: item.total_checks,
                    PassPercent: typeof item.percentage_pass === 'number'
                        ? item.percentage_pass.toFixed(2) + "%"
                        : item.percentage_pass,
                }));
                setQcDataRows(rows);
            } catch (e) {
                console.error("Error loading qc_summary_results.json:", e);
                setQcDataRows([]);
            }
        })();
    }, []);

    // Variable summary
    useEffect(() => {
        (async () => {
            try {
                const data: any[] = await loadJson("variable_summary_results.json");
                if (!Array.isArray(data)) throw new Error("Not an array");
                const rows = data.map((item, i) => withTwoPercents({
                    id: item.Variable || i,
                    Variable: item.Variable,
                    isPassed: item.Failed === 0,
                    Pass: item.Passed,
                    Fail: item.Failed,
                    Amount: item.Total,
                    PassPercent: typeof item.PercentagePass === 'number'
                        ? item.PercentagePass.toFixed(2) + "%"
                        : item.PercentagePass,
                }));
                setVariableDataRows(rows);
            } catch (e) {
                console.error("Error loading variable_summary_results.json:", e);
                setVariableDataRows([]);
            }
        })();
    }, []);

    // Datasource missingness
    useEffect(() => {
        (async () => {
            try {
                const data: any[] = await loadJson("datasource_missingness_results.json");
                if (!Array.isArray(data)) throw new Error("Not an array");
                const rows = data.map((item, i) => withTwoPercents({
                    id: item.Datasource || i,
                    Datasource: item.Datasource,
                    isPassed: item.Missing === 0,
                    Present: item.Present,
                    Missing: item.Missing,
                    Total: item.Total,
                    MissingPercent: typeof item.MissingPercent === 'number'
                        ? item.MissingPercent.toFixed(2) + "%"
                        : item.MissingPercent,
                }));
                setDatasourceDataRows(rows);
            } catch (e) {
                console.error("Error loading datasource_missingness_results.json:", e);
                setDatasourceDataRows([]);
            }
        })();
    }, []);

    // Importance group summary
    useEffect(() => {
        (async () => {
            try {
                const data: any[] = await loadJson("importance_group_summary_results.json");
                if (!Array.isArray(data)) throw new Error("Not an array");
                const rows = data.map((rec, i) => withTwoPercents({
                    id: rec.Group || i,
                    Group: rec.Group === "M" ? "High"
                        : rec.Group === "R" ? "Medium"
                            : rec.Group === "O" ? "Low"
                                : rec.Group,
                    isPassed: rec.Failed === 0,
                    Passed: rec.Passed,
                    Failed: rec.Failed,
                    Total: rec.Total,
                    PassPercent: typeof rec.PercentagePass === 'number'
                        ? rec.PercentagePass.toFixed(2) + "%"
                        : rec.PercentagePass,
                }));
                setImportanceDataRows(rows);
            } catch (e) {
                console.error("Error loading importance_group_summary_results.json:", e);
                setImportanceDataRows([]);
            }
        })();
    }, []);

    // Phase missingness summary
    useEffect(() => {
        (async () => {
            try {
                const data: any[] = await loadJson("phase_missingness_results.json");
                if (!Array.isArray(data)) throw new Error("Not an array");
                const rows = data.map((rec, i) => withTwoPercents({
                    id: rec.Phase || i,
                    Phase: rec.Phase,
                    Present: rec.Present,
                    Missing: rec.Missing,
                    Total: rec.Total,
                    MissingPercent: typeof rec.MissingPercent === 'number'
                        ? rec.MissingPercent.toFixed(2) + "%"
                        : rec.MissingPercent,
                }));
                setPhaseDataRows(rows);
            } catch (e) {
                console.error("Error loading phase_missingness_results.json:", e);
                setPhaseDataRows([]);
            }
        })();
    }, []);

    // Patient importance summary
    useEffect(() => {
        (async () => {
            try {
                const raw: any[] = await loadJson("patient_importance_summary_results.json");
                if (!Array.isArray(raw)) throw new Error("Not an array");
                const orderMap: Record<string, number> = { M: 0, R: 1, O: 2 };
                const label = (g: string) => g === "M" ? "High"
                    : g === "R" ? "Medium"
                        : g === "O" ? "Low"
                            : g;
                const rows = raw.map((r, i) => withTwoPercents({
                    id: `${r.PatientID}-${r.Group}-${i}`,
                    PatientID: r.PatientID,
                    Group: label(r.Group),
                    _code: r.Group,
                    Passed: r.Passed,
                    Failed: r.Failed,
                    Total: r.Total,
                    PassPercent: r.PassPercent,
                })).sort((a, b) =>
                    a.PatientID !== b.PatientID
                        ? Number(a.PatientID) - Number(b.PatientID)
                        : orderMap[a._code] - orderMap[b._code]
                );
                let last = "";
                rows.forEach(r => { if (r.PatientID !== last) { r.__firstOfPatient = true; last = r.PatientID; } });
                setPatientImpRows(rows);
            } catch (e) {
                console.error("Error loading patient_importance_summary_results.json:", e);
                setPatientImpRows([]);
            }
        })();
    }, []);

    // Patient phase summary
    useEffect(() => {
        (async () => {
            try {
                const raw: any[] = await loadJson("patient_phase_summary_results.json");
                if (!Array.isArray(raw)) throw new Error("Not an array");
                const PHASES = ["Diagnosis", "Progression", "Recurrence"];
                const byPid: Record<string, any> = {};
                raw.forEach(r => { const pid = String(r.PatientID); if (!byPid[pid]) byPid[pid] = {}; byPid[pid][r.Phase] = r; });
                const rows: any[] = [];
                Object.keys(byPid).sort((a, b) => Number(a) - Number(b)).forEach(pid => {
                    PHASES.forEach(phase => {
                        const src = byPid[pid][phase] ?? { Total: 0, Missing: 0 };
                        rows.push(withTwoPercents({
                            id: `${pid}_${phase}`,
                            PatientID: pid,
                            Phase: phase,
                            Present: src.Total - src.Missing,
                            Missing: src.Missing,
                            Total: src.Total,
                            MissingPercent: src.Total ? ((src.Missing / src.Total) * 100).toFixed(1) + "%" : "0.0%",
                        }));
                    });
                });
                let last = "";
                rows.forEach(r => { if (r.PatientID !== last) { r.__firstOfPatient = true; last = r.PatientID; } });
                setPhasePatientRows(rows);
            } catch (e) {
                console.error("Error loading patient_phase_summary_results.json:", e);
                setPhasePatientRows([]);
            }
        })();
    }, []);

    // Phase → Entity timeline data (and summary rows)
    useEffect(() => {
        (async () => {
            try {
                const raw: any[] = await loadJson("patient_entity_phase_results.json");
                const byPid: Record<string, any[]> = {};
                raw.forEach(r => {
                    const pid = String(r.PatientID);
                    if (!byPid[pid]) byPid[pid] = [];
                    let row = byPid[pid].find(x => x.Entity === r.Entity);
                    if (!row) {
                        row = { id: `${r.Entity}-${pid}`, Entity: r.Entity, Diagnosis: null, Progression: null, Recurrence: null };
                        byPid[pid].push(row);
                    }
                    row[r.Phase as "Diagnosis" | "Progression" | "Recurrence"] = r.Complete;
                    row[`Missing${r.Phase}`] = r.Missing;
                });
                patientTimeline.current = byPid;
                const summary = Object.entries(byPid).map(([pid, rows]) => {
                    const collapse = (phase: string) => {
                        const vals = rows.map(r => r[phase]);
                        if (!vals.length) return null;
                        const someTrue = vals.includes(true);
                        const someFalse = vals.includes(false);
                        if (someTrue && someFalse) return "mixed";
                        return someTrue;
                    };
                    return {
                        id: pid,
                        PatientID: pid,
                        Diagnosis: collapse("Diagnosis"),
                        Progression: collapse("Progression"),
                        Recurrence: collapse("Recurrence"),
                    };
                });
                setPhaseRows(summary as any[]);
            } catch (e) {
                console.error("Error loading patient_entity_phase_results.json:", e);
                setPhaseRows([]);
            }
        })();
    }, []);

    // CSV download utils
    const rowsToCsv = (rows: any[], cols: { field: string }[]) => {
        if (!rows.length) return "";
        const headers = cols.map(c => `"${c.field}"`).join(",");
        const body = rows.map(r => cols.map(c => `"${String(r[c.field] ?? "")}"`).join(",")).join("\n");
        return `${headers}\n${body}`;
    };

    const handleDownload = () => {
        const tabMap: Record<number, { rows: any[]; cols: any[] }> = {
            0: { rows: summaryData, cols: columns },
            1: { rows: qcDataRows, cols: columnsByQC },
            2: { rows: patientDataRows, cols: columnsByPatient },
            3: { rows: variableDataRows, cols: columnsByVariable },
            4: { rows: datasourceDataRows, cols: columnsByDatasource },
            5: { rows: importanceDataRows, cols: columnsByImportance },
            6: { rows: phaseDataRows, cols: columnsByPhase },
            7: { rows: patientImpRows, cols: columnsByImpPat },
            8: { rows: phasePatientRows, cols: columnsPhasePatient },
        };
        const { rows, cols } = tabMap[activeTab] ?? {};
        if (!rows?.length) return;
        const csv = rowsToCsv(rows, cols);
        const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
        const tabNames = [
            "summary", "by_qc", "by_patient_id", "by_variable",
            "by_datasource", "by_importance", "by_phase",
            "importance_patient", "phase_patient",
        ];
        const today = new Date().toISOString().slice(0, 10);
        saveAs(blob, `${tabNames[activeTab]}_${today}.csv`);
    };

    // Column definitions
    const basePctCols: GridColDef[] = [
        { field: "MissingPercent", headerName: "% Missing", width: 110 },
        { field: "PassPercent", headerName: "% Complete", width: 110 }
    ];

    const columns: GridColDef[] = [
        { field: "category", headerName: "", width: 150 },
        { field: "Pass", headerName: "Total Pass", width: 150 },
        { field: "Fail", headerName: "Total Fail", width: 150 },
        { field: "Amount", headerName: "Total", width: 150 },
        ...basePctCols,
    ];

    const columnsByQC: GridColDef[] = [
        { field: "QC", headerName: "QC", width: 150 },
        {
            field: "isPassed", headerName: "", width: 50,
            renderCell: (params) => params.value
                ? <CheckCircleIcon color="success" />
                : <CancelIcon color="error" />
        },
        { field: "Pass", headerName: "Passed", width: 100 },
        { field: "Fail", headerName: "Failed", width: 100 },
        { field: "Amount", headerName: "Total", width: 100 },
        ...basePctCols,
    ];

    const columnsByPatient: GridColDef[] = [
        { field: "PatientID", headerName: "PID", width: 150 },
        {
            field: "isPassed", headerName: "", width: 50,
            renderCell: (params) => params.value
                ? <CheckCircleIcon color="success" />
                : <CancelIcon color="error" />
        },
        { field: "Pass", headerName: "Passed", width: 100 },
        { field: "Fail", headerName: "Failed", width: 100 },
        { field: "Amount", headerName: "Total", width: 100 },
        ...basePctCols,
    ];

    const columnsByVariable: GridColDef[] = [
        { field: "Variable", headerName: "Variable", width: 200 },
        {
            field: "isPassed", headerName: "", width: 50,
            renderCell: (params) => params.value
                ? <CheckCircleIcon color="success" />
                : <CancelIcon color="error" />
        },
        { field: "Pass", headerName: "Passed", width: 100 },
        { field: "Fail", headerName: "Failed", width: 100 },
        { field: "Amount", headerName: "Total", width: 100 },
        ...basePctCols,
    ];

    const columnsByDatasource: GridColDef[] = [
        { field: "Datasource", headerName: "Source", width: 180 },
        {
            field: "isPassed", headerName: "", width: 50,
            renderCell: (params) => params.value
                ? <CheckCircleIcon color="success" />
                : <CancelIcon color="error" />
        },
        { field: "Present", headerName: "Present", width: 110 },
        { field: "Missing", headerName: "Missing", width: 110 },
        { field: "Total", headerName: "Total", width: 110 },
        ...basePctCols,
    ];

    const columnsByImportance: GridColDef[] = [
        { field: "Group", headerName: "Group", width: 130 },
        {
            field: "isPassed", headerName: "", width: 50,
            renderCell: (params) => params.value
                ? <CheckCircleIcon color="success" />
                : <CancelIcon color="error" />
        },
        { field: "Passed", headerName: "Passed", width: 110 },
        { field: "Failed", headerName: "Failed", width: 110 },
        { field: "Total", headerName: "Total", width: 110 },
        ...basePctCols,
    ];

    const columnsByPhase: GridColDef[] = [
        { field: "Phase", headerName: "Phase", width: 140 },
        { field: "Present", headerName: "Present", width: 110 },
        { field: "Missing", headerName: "Missing", width: 110 },
        { field: "Total", headerName: "Total", width: 110 },
        ...basePctCols,
    ];

    const columnsByImpPat: GridColDef[] = [
        { field: "PatientID", headerName: "PID", width: 110 },
        { field: "Group", headerName: "Group", width: 110 },
        { field: "Passed", headerName: "Passed", width: 100 },
        { field: "Failed", headerName: "Failed", width: 100 },
        { field: "Total", headerName: "Total", width: 100 },
        ...basePctCols,
    ];

    const columnsPhasePatient: GridColDef[] = [
        { field: "PatientID", headerName: "PID", width: 120 },
        { field: "Phase", headerName: "Phase", width: 140 },
        { field: "Present", headerName: "Present", width: 110 },
        { field: "Missing", headerName: "Missing", width: 110 },
        { field: "Total", headerName: "Total", width: 110 },
        ...basePctCols,
    ];

    const columnsPhaseEntity: GridColDef[] = [
        { field: "Entity", headerName: "Entity", width: 200 },
        ...["Diagnosis", "Progression", "Recurrence"].map(phase => ({
            field: phase,
            headerName: phase[0],
            width: 80,
            sortable: false,
            renderCell: (params) =>
                params.value
                    ? <CheckCircleIcon color="success" />
                    : <CancelIcon color="error" />
        })),
    ];

    const phaseColumns = useMemo(() => [
        { field: "PatientID", headerName: "Patient", flex: 1, minWidth: 160 },
        ...["Diagnosis", "Progression", "Recurrence"].map(ph => ({
            field: ph,
            headerName: ph[0],
            width: 90,
            sortable: false,
            renderCell: ({ value }: any) => {
                if (value === null) return <RemoveCircleOutlineIcon color="disabled" />;
                if (value === "mixed") return <WarningAmberIcon color="warning" />;
                return value
                    ? <CheckCircleIcon color="success" />
                    : <CancelIcon color="error" />;
            }
        }))
    ], []);

    return (
        <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
            <Stack spacing={2} sx={{ flex: 1, overflow: "auto" }}>
                <Stack spacing={2} sx={{ alignItems: "center", justifyContent: "center", flexGrow: 1 }}>
                    <Typography level="h2">The report about your data:</Typography>
                    <Tabs variant="outlined" value={activeTab} onChange={(_, v) => setActiveTab(Number(v))} sx={{ width: "auto", borderRadius: "lg", boxShadow: "sm", overflow: "auto" }}>
                        <TabList disableUnderline tabFlex={1} sx={{
                            [`& .${tabClasses.root}`]: {
                                fontSize: "sm", fontWeight: "lg",
                                [`&[aria-selected=\"true\"]`]: { color: "primary.500", bgcolor: "background.surface" },
                                [`&.${tabClasses.focusVisible}`]: { outlineOffset: "-4px" }
                            }
                        }}>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>Summary</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>By QC</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>By patient ID</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>By Variable</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>By datasource</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>By importance</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>By phase</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>Importance ⇢ Patient</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>Phase ⇢ Patient</Tab>
                        </TabList>
                        <TabPanel value={0}><DataGrid rows={summaryData} columns={columns} /></TabPanel>
                        <TabPanel value={1}><DataGrid rows={qcDataRows} columns={columnsByQC} /></TabPanel>
                        <TabPanel value={2}>
                            <Stack spacing={1}><Typography level="body-sm" sx={{ color: "text.tertiary" }}>👉 Click a patient row to inspect the failed quality checks</Typography>
                                <DataGrid rows={patientDataRows} columns={columnsByPatient} onRowClick={({ row }) => {
                                    const pid = String(row.PatientID);
                                    setFailModalPatient(pid);
                                    setFailModalRows(patientFailures.current.get(pid) ?? []);
                                    setFailModalOpen(true);
                                }} />
                            </Stack>
                        </TabPanel>
                        <TabPanel value={3}><DataGrid rows={variableDataRows} columns={columnsByVariable} /></TabPanel>
                        <TabPanel value={4}><DataGrid rows={datasourceDataRows} columns={columnsByDatasource} /></TabPanel>
                        <TabPanel value={5}><DataGrid rows={importanceDataRows} columns={columnsByImportance} /></TabPanel>
                        <TabPanel value={6}><DataGrid rows={phaseRows} columns={phaseColumns} onRowClick={({ row }) => {
                            const pid = String(row.PatientID);
                            setTimelinePID(pid);
                            setTimelineRows(patientTimeline.current[pid] ?? []);
                            setTimelineOpen(true);
                        }} /></TabPanel>
                        <TabPanel value={7}><DataGrid rows={patientImpRows} columns={columnsByImpPat} getRowClassName={({ row }) =>
                            `${Number(row.PatientID) % 2 === 0 ? "pid-even" : "pid-odd"} ${row.__firstOfPatient ? "patient-divider" : ""}`.trim()
                        } sx={{
                            "& .pid-even": { bgcolor: "#f5f5f5" },
                            "& .pid-odd": { bgcolor: "background.surface" },
                            "& .patient-divider > .MuiDataGrid-cell": { borderTop: "2px solid", borderColor: "divider" }
                        }} /></TabPanel>
                        <TabPanel value={8}><DataGrid rows={phasePatientRows} columns={columnsPhasePatient} getRowClassName={({ row }) =>
                            `${Number(row.PatientID) % 2 === 0 ? "pid-even" : "pid-odd"} ${row.__firstOfPatient ? "patient-divider" : ""}`.trim()
                        } sx={{
                            "& .pid-even": { bgcolor: "#f5f5f5" },
                            "& .pid-odd": { bgcolor: "background.surface" },
                            "& .patient-divider > .MuiDataGrid-cell": { borderTop: "2px solid", borderColor: "divider" }
                        }} /></TabPanel>
                    </Tabs>
                </Stack>
            </Stack>
            <PatientTimeline open={timelineOpen} onClose={() => setTimelineOpen(false)} patientId={timelinePID} rows={timelineRows} />
        </Box>
    );
}
