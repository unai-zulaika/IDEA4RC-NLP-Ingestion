import React, { useEffect, useState, useRef, useMemo, useCallback } from "react";
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
import dimensionSummary from "/data/results/dimension_summary_results.json";
import patientSummary from "/data/results/patient_summary_results.json";
import qcSummary from "/data/results/qc_summary_results.json";
import variableSummary from "/data/results/variable_summary_results.json";
import datasourceMissingness from "/data/results/datasource_missingness_results.json";
import importanceGroupSummary from "/data/results/importance_group_summary_results.json";
import phaseMissingness from "/data/results/phase_missingness_results.json";
import patientImportanceSummary from "/data/results/patient_importance_summary_results.json";
import patientPhaseSummary from "/data/results/patient_phase_summary_results.json";
import patientEntityPhaseResults from "/data/results/patient_entity_phase_results.json";
import inSetPerInstance from "/data/results/in_set_expectation_per_instance.json";
import inSetResults from "/data/results/in_set_expectation_results.json";
import datatypeRows from "/data/results/datatype_expectation_results.json";
import inSetRows from "/data/results/in_set_expectation_per_instance.json";
import phaseEntityResults from "/data/results/phase_entity_results.json";
import { read, utils } from "xlsx";

// util: format a number → "12.34 %" or fallback to "0.00 %"
const pct = (v: number | undefined | null) =>
    typeof v === "number" && !Number.isNaN(v) ? v.toFixed(2) + "%" : "0.00%";

const num = (x: any, d = 0) => (typeof x === "number" && !Number.isNaN(x) ? x : d);

type AnyRow = Record<string, any>;

type DatatypeRow = {
    Variable: string;
    ExpectedType?: string;
    CurrentDatatype?: string;
    DatatypeCorrect?: boolean;
    PatientsFailed?: string[];
    Sources?: string[];
    // ...other fields ignored for popup...
} & AnyRow;

type InSetRow = {
    Variable: string;
    Entity?: string;
    EntityInstance?: string | null;
    ExpectedSet?: string[];
    ObservedValues?: string[];
    UnexpectedValues?: string[];
    Failed?: boolean;
    PatientsAffected?: string[];
    // ...other fields ignored for popup...
} & AnyRow;

function Popup(props: {
    open: boolean;
    title?: string;
    onClose: () => void;
    children?: React.ReactNode;
}) {
    if (!props.open) return null;
    return (
        <div
            style={{
                position: "fixed",
                inset: 0,
                background: "rgba(0,0,0,0.35)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                zIndex: 9999,
            }}
            onClick={props.onClose}
        >
            <div
                style={{
                    minWidth: 360,
                    maxWidth: "80vw",
                    maxHeight: "80vh",
                    overflow: "auto",
                    background: "#fff",
                    borderRadius: 8,
                    boxShadow: "0 8px 28px rgba(0,0,0,0.25)",
                    padding: 16,
                }}
                onClick={(e) => e.stopPropagation()}
            >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <h3 style={{ margin: 0 }}>{props.title || "Details"}</h3>
                    <button onClick={props.onClose} aria-label="Close">✕</button>
                </div>
                <div style={{ marginTop: 12 }}>{props.children}</div>
            </div>
        </div>
    );
}

// Utility to fetch JSON from public/data folder
async function loadJson<T = any>(fileName: string): Promise<T> {
    const res = await fetch(`/data/results/${fileName}`);
    if (!res.ok) throw new Error(`Failed to fetch ${fileName}: ${res.statusText}`);
    return res.json();
}

export default function Results() {
    const [summaryData, setSummaryData] = React.useState([]);
    const [patientDataRows, setPatientDataRows] = React.useState([]);
    const [qcDataRows, setQcDataRows] = React.useState([]);
    const [variableDataRows, setVariableDataRows] = React.useState<any[]>([]);
    const [datasourceDataRows, setDatasourceDataRows] = React.useState<any[]>([]);
    const [importanceDataRows, setImportanceDataRows] = React.useState<any[]>([]);
    const [datatypeGridRows, setDatatypeRows] = React.useState<any[]>([]); // renamed to avoid import name clash
    const [patientImpRows, setPatientImpRows] = React.useState<any[]>([]);
    const [phasePatientRows, setPhasePatientRows] = React.useState<any[]>([]);
    const [phaseEntityRows, setPhaseEntityRows] = React.useState<any[]>([]);

    // ─────────────────────────────────────────────────────────────
    const [phaseRows, setPhaseRows] = React.useState<any[]>([]);   // ← add this

    const [timelineOpen, setTimelineOpen] = React.useState(false);
    const [timelineRows, setTimelineRows] = React.useState<any[]>([]);
    const [timelinePID, setTimelinePID] = React.useState<string>("");
    // keeps a map  { patientId → entity-level rows }
    const patientTimeline = React.useRef<Record<string, any[]>>({});
    // ─────────────────────────────────────────────────────────────

    // ── NEW: In-Set QC state ─────────────────────────────────────
    const [inSetPerInstanceRows, setInSetPerInstanceRows] = React.useState<any[]>([]);
    const [inSetResultsSummary, setInSetResultsSummary] = React.useState<any[]>([]);
    // Columns for the new tab (same structure you use for other data tables)
    const inSetColumns = React.useMemo(
        () => [
            { field: "variable", headerName: "Variable", flex: 1.5 },
            { field: "entity", headerName: "Entity", flex: 1 },
            { field: "entityInstance", headerName: "Entity ID", flex: 0.8 },
            {
                field: "passed",
                headerName: "",
                width: 60,
                align: "center",
                headerAlign: "center",
                sortable: true,
                renderCell: (p: any) =>
                    p.value ? <CheckCircleIcon color="success" /> : <CancelIcon color="error" />,
            },
            { field: "expected", headerName: "Expected Codes", flex: 1.6 },
            { field: "error", headerName: "Unexpected Codes", flex: 1.6 },
        ],
        []
    );

    const phaseColumns = React.useMemo(
        () => [
            { field: "PatientID", headerName: "Patient", flex: 1, minWidth: 160 },
            ...["Diagnosis", "Progression", "Recurrence"].map((ph) => ({
                field: ph,
                headerName: ph[0],          // D / P / R
                width: 90,
                sortable: false,
                renderCell: (p: any) => {
                    const val = p.value as boolean | "mixed" | null;
                    if (val === null) return <RemoveCircleOutlineIcon color="disabled" />;
                    if (val === "mixed") return <WarningAmberIcon color="warning" />;       // ⚠️
                    return val
                        ? <CheckCircleIcon color="success" />                                  // ✔
                        : <CancelIcon color="error" />;                                        // ✗
                },
            })),
        ],
        []
    );


    // Load and map the JSONs
    React.useEffect(() => {
        try {
            const perInstance: any[] = (inSetPerInstance as any) ?? [];
            const perInstanceMapped = perInstance.map((d, i) => ({
                id: `${d.Variable}-${d.Entity}-${d.EntityInstance ?? i}`,
                variable: d.Variable,
                entity: d.Entity,
                entityInstance: d.EntityInstance ?? "-",
                // was: passed only; add a failed flag too
                passed: d.Failed === false,
                failed: !!d.Failed,
                expected: Array.isArray(d.ExpectedSet) ? d.ExpectedSet.join(", ") : "",
                // Observed unexpected values (only present when there is a failure)
                error: Array.isArray(d.UnexpectedValues) ? d.UnexpectedValues.join(", ") : "",
            }));
            setInSetPerInstanceRows(perInstanceMapped);

            const results: any[] = (inSetResults as any) ?? [];
            setInSetResultsSummary(results);
        } catch (err) {
            console.error("Failed to load In-Set QC data", err);
        }
    }, []);
    // ─────────────────────────────────────────────────────────────

    // ── modal ────────────────────────────────────────────────────────────────
    const [failModalOpen, setFailModalOpen] = React.useState(false);
    const [failModalRows, setFailModalRows] = React.useState<any[]>([]);
    const [failModalPatient, setFailModalPatient] = React.useState<string>("");
    const [phaseDataRows, setPhaseDataRows] = React.useState<any[]>([]);
    const [activeTab, setActiveTab] = React.useState<number>(0);

    const [matrixRows, setMatrixRows] = React.useState<any[]>([]);
    const [matrixCols, setMatrixCols] = React.useState<any[]>([]);

    /** Decide which dataset is on-screen & trigger the download */
    /** Build a CSV string from rows + column definitions */
    const rowsToCsv = (rows: any[], cols: { field: string }[]) => {
        if (!rows?.length) return "";
        const hdr = cols.map(c => `"${c.field}"`).join(",");
        const body = rows
            .map(r => cols.map(c => `"${String(r[c.field] ?? "")}"`).join(","))
            .join("\n");
        return `${hdr}\n${body}`;
    };


    // columns for the modal grid
    const columnsFail: GridColDef[] = [
        { field: "Variable", headerName: "Variable", width: 180 },
        { field: "QC", headerName: "QC", width: 220 },
        { field: "BadValue", headerName: "Value", width: 140 },
    ];

    const columnsByDatasourceCsv: GridColDef[] = [
        { field: "Datasource", headerName: "Datasource", width: 170 },
        { field: "Section", headerName: "Section", width: 120 },     // "overall" | "dimension"
        { field: "Dimension", headerName: "Dimension", width: 140 }, // Plausibility | Conformance | Completeness | ""
        { field: "Scope", headerName: "Scope", width: 140 },         // "checks" | "variables" | "patients"
        { field: "Unit", headerName: "Unit", width: 180 },           // kept for compatibility
        { field: "Passed", headerName: "Passed", width: 120 },
        { field: "Failed", headerName: "Failed", width: 120 },
        { field: "Total", headerName: "Total", width: 120 },
        { field: "CompletePercent", headerName: "Complete %", width: 140 },
        { field: "MissingPercent", headerName: "Missing %", width: 140 },
    ];


    /* ---------------------------------------------------------------------- */
    /*  Build patient ➜ [{Variable, QC, Value, Entity}] lookup once on mount  */
    /* ---------------------------------------------------------------------- */
    const patientFailures = React.useRef<Map<string, any[]>>(new Map());

    /** ensure a row always has both MissingPercent & PassPercent */
    const withTwoPercents = (row: any) => {
        if (row.MissingPercent == null && row.PassPercent != null) {
            // only “complete/pass” present  → derive missingness
            const v = parseFloat(String(row.PassPercent).replace("%", ""));
            row.MissingPercent = (100 - v).toFixed(1) + "%";
        } else if (row.PassPercent == null && row.MissingPercent != null) {
            // only “missing” present → derive completeness
            const v = parseFloat(String(row.MissingPercent).replace("%", ""));
            row.PassPercent = (100 - v).toFixed(1) + "%";
        }
        return row;
    };

    // React.useEffect(() => {
    //     // list every JSON file whose name starts with "results"
    //     const allFiles: string[] = window.electron.data.list_files()
    //         .filter(f => f.startsWith("results") && f.endsWith(".json"));

    //     const map = new Map<string, any[]>();            // PatientID → rows[]
    //     const seen = new Set<string>();                  // PatientID|Var|QC|Val|Ent

    //     allFiles.forEach(fname => {
    //         const doc = window.electron.data.load_json(fname);
    //         Object.values(doc.run_results ?? {}).forEach((run: any) => {
    //             const entityName = fname === "results.json"
    //                 ? "non_repeatable"
    //                 : fname.replace(/^results_|\..+$/g, "");

    //             run.validation_result?.results?.forEach((res: any) => {
    //                 if (res.success) return;                         // only failures
    //                 const varName = res.expectation_config?.kwargs?.column ?? "Table-level";
    //                 const qcName = res.expectation_config?.expectation_type ?? "N/A";

    //                 (res.result?.partial_unexpected_index_list ?? []).forEach((idxObj: any) => {
    //                     const pid = String(idxObj.patient_id ?? idxObj.PatientID ?? "UNKNOWN");
    //                     const bad = String(idxObj[varName] ?? idxObj.value ?? "NULL");

    //                     const dedupKey = `${pid}|${varName}|${qcName}|${bad}|${entityName}`;
    //                     if (seen.has(dedupKey)) return;                // ← skip duplicate
    //                     seen.add(dedupKey);

    //                     const row = {
    //                         id: dedupKey,                         // guaranteed unique
    //                         PatientID: pid,
    //                         Variable: varName,
    //                         QC: qcName,
    //                         BadValue: bad,
    //                         Entity: entityName,
    //                     };
    //                     if (!map.has(pid)) map.set(pid, []);
    //                     map.get(pid)!.push(row);
    //                 });
    //             });
    //         });
    //     });
    //     patientFailures.current = map;

    // }, []);

    // MODIFIED useEffect for summaryData (dimension_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!dimensionSummary || typeof dimensionSummary !== "object") {
                    setSummaryData([]);
                    return;
                }
                const categoriesInOrder = ["Plausibility", "Conformance", "Completeness", "Total"];
                let idCounter = 1;
                const transformedData = categoriesInOrder
                    .map((categoryName) => {
                        const data = (dimensionSummary as any)[categoryName];
                        return data
                            ? withTwoPercents({
                                id: idCounter++,
                                category: categoryName,
                                Pass: data.Passed,
                                Fail: data.Failed,
                                Amount: data.Total,
                                PassPercent: data.PercentagePass,
                            })
                            : null;
                    })
                    .filter(Boolean) as any[];
                setSummaryData(transformedData);
            } catch (error) {
                console.error("Error loading dimension_summary_results.json:", error);
                setSummaryData([]);
            }
        })();
    }, []); // Empty dependency array to run once on mount

    // MODIFIED useEffect for patientDataRows (patient_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(patientSummary)) throw new Error("Not an array");
                const rows = patientSummary.map((item: any, i: number) =>
                    withTwoPercents({
                        id: item.PatientID || i,
                        PatientID: item.PatientID,
                        isPassed: item.Failed === 0,
                        Pass: item["Number of Passed Tests"],
                        Fail: item.Failed,
                        Amount: item.Total,
                        PassPercent:
                            typeof item.PercentagePass === "number"
                                ? item.PercentagePass.toFixed(2) + "%"
                                : item.PercentagePass,
                    })
                );
                setPatientDataRows(rows);
            } catch (e) {
                console.error("Error loading patient_summary_results.json:", e);
                setPatientDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for qcDataRows (qc_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(qcSummary)) {
                    setQcDataRows([]);
                    return;
                }
                const rows: any[] = [];
                qcSummary.forEach((item: any, idx: number) => {
                    const addRow = (unit: string, passed: number, failed: number, completePct: number, missingPct: number) =>
                        rows.push({
                            id: `${idx}-${unit}`,
                            QC_name: item.QC ?? item.ge_name,
                            Unit: unit,
                            Passed: passed,
                            Failed: failed,
                            CompletePercent: pct(completePct),
                            MissingPercent: pct(missingPct),
                        });
                    addRow("found variables", item.passed_variables, item.failed_variables, item.complete_variables_percentage, item.missing_variables_percentage);
                    addRow("full dm variables", item.full_dm_passed_variables, item.full_dm_failed_variables, item.full_dm_complete_variables_percentage, item.full_dm_missing_variables_percentage);
                    addRow("patients", item.passed_patients, item.failed_patients, item.complete_patients_percentage, item.missing_patients_percentage);
                    addRow("performed quality checks", item.passed_quality_checks, item.failed_quality_checks, item.quality_checks_percentage_pass, item.quality_checks_percentage_fail);
                });
                setQcDataRows(rows);
            } catch (error) {
                console.error("Error loading qc_summary_results.json:", error);
                setQcDataRows([]);
            }
        })();
    }, []); // Empty dependency array to run once on mount

    // MODIFIED useEffect for variableDataRows (variable_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(variableSummary)) {
                    setVariableDataRows([]);
                    return;
                }
                const rows = variableSummary.map((item: any, idx: number) =>
                    withTwoPercents({
                        id: item.Variable ?? idx,
                        Variable: item.Variable,
                        isPassed: item.Failed === 0,
                        Pass: item.Passed,
                        Fail: item.Failed,
                        Amount: item.Total,
                        PassPercent:
                            typeof item.PercentagePass === "number"
                                ? item.PercentagePass.toFixed(2) + "%"
                                : item.PercentagePass,
                    })
                );
                setVariableDataRows(rows);
            } catch (e) {
                console.error("Error loading variable_summary_results.json:", e);
                setVariableDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for datasourceDataRows (datasource_missingness_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(datasourceMissingness)) {
                    setDatasourceDataRows([]);
                    return;
                }
                const normalRows: any[] = [];
                const unknownRows: any[] = [];
                const maybePush = (row: any, ds: string) =>
                    (ds.toLowerCase() === "unknown_source" ? unknownRows : normalRows).push(row);

                datasourceMissingness.forEach((item: any, idx: number) => {
                    const addRow = (unit: string, passed: number, failed: number, completePct: number, missingPct: number, cls?: string) =>
                        maybePush(
                            {
                                id: `${item.Datasource}-${unit}-${idx}`,
                                Datasource: item.Datasource,
                                Unit: unit,
                                Passed: passed,
                                Failed: failed,
                                CompletePercent: pct(completePct),
                                MissingPercent: pct(missingPct),
                                __cls: cls,
                            },
                            item.Datasource
                        );
                    addRow("variables", item.passed_variables, item.failed_variables, item.complete_variables_percentage, item.missing_variables_percentage);
                    addRow("patients", item.passed_patients, item.failed_patients, item.complete_patients_percentage, item.missing_patients_percentage);
                    addRow("performed quality checks", item.passed_quality_checks, item.failed_quality_checks, item.quality_checks_percentage_passed, item.quality_checks_percentage_fail);
                    const dims = item.dimensions || {};
                    ["Plausibility", "Conformance", "Completeness"].forEach((dimName) => {
                        const d = dims[dimName];
                        if (!d) return;
                        const vt = Number(d.VarsTotal || 0);
                        const vp = Number(d.VarsPassed || 0);
                        const vf = Number(d.VarsFailed || 0);
                        const compVarPct = vt ? (vp / vt) * 100 : 0;
                        const missVarPct = vt ? (vf / vt) * 100 : 0;
                        addRow(`Variables ${dimName}`, vp, vf, compVarPct, missVarPct, "dim-vars");
                    });
                });
                setDatasourceDataRows([...normalRows, ...unknownRows]);
            } catch (e) {
                console.error("Error loading datasource_missingness_results.json:", e);
                setDatasourceDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for importanceDataRows (importance_group_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(importanceGroupSummary)) throw new Error("Not an array");
                const rows: any[] = [];
                const nice = (g: string) => (g === "M" ? "High" : g === "R" ? "Medium" : g === "O" ? "Low" : g);
                importanceGroupSummary.forEach((rec: any, idx: number) => {
                    const add = (unit: string, passed: number, failed: number, complete: number, missing: number) =>
                        rows.push({
                            id: `${rec.Group}-${unit}-${idx}`,
                            Group: nice(rec.Group),
                            Unit: unit,
                            Passed: passed,
                            Failed: failed,
                            CompletePercent: pct(complete),
                            MissingPercent: pct(missing),
                        });
                    const totVar = rec.total_variables || 0;
                    const passVar = rec.passed_variables || 0;
                    add("found variables", passVar, rec.failed_variables || 0, totVar ? (passVar / totVar) * 100 : 0, totVar ? 100 - (passVar / totVar) * 100 : 0);
                    add("full dm variables", rec.full_dm_passed_variables, rec.full_dm_failed_variables, rec.full_dm_complete_variables_percentage, rec.full_dm_missing_variables_percentage);
                    const totPat = rec.total_patients || 0;
                    const passPat = rec.passed_patients || 0;
                    add("patients", passPat, rec.failed_patients || 0, totPat ? (passPat / totPat) * 100 : 0, totPat ? 100 - (passPat / totPat) * 100 : 0);
                    add("performed quality checks", rec.passed_quality_checks, rec.failed_quality_checks, rec.quality_checks_percentage_passed, rec.quality_checks_percentage_fail);
                });
                setImportanceDataRows(rows);
            } catch (e) {
                console.error("Error loading importance_group_summary_results.json:", e);
                setImportanceDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for datatypeGridRows (datatype_expectation_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(datatypeRows)) throw new Error("Not an array");
                const filtered = datatypeRows.filter((r: any) => String(r?.CurrentDatatype ?? "") !== "Missing");
                const rows = filtered.map((r: any, idx: number) => ({
                    id: r.Variable ?? idx,
                    Variable: r.Variable,
                    ExpectedType: r.ExpectedType,
                    CurrentDatatype: r.CurrentDatatype ?? null,
                    DatatypeCorrect: !!r.DatatypeCorrect,
                    PercentagePass: typeof r.PercentagePass === "number" ? r.PercentagePass.toFixed(2) + "%" : r.PercentagePass ?? "0.00%",
                    PatientsFailed: Array.isArray(r.PatientsFailed) ? r.PatientsFailed : [],
                    Sources: Array.isArray(r.Sources) ? r.Sources : [],
                }));
                setDatatypeRows(rows);
            } catch (e) {
                console.error("Error loading datatype_expectation_results.json", e);
                setDatatypeRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for phaseDataRows (phase_missingness_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(phaseMissingness)) throw new Error("Not an array");
                const rows = phaseMissingness.map((rec: any, idx: number) => ({
                    id: rec.Phase ?? idx,
                    Phase: rec.Phase,
                    MissingPercent_Variable: rec.MissingPercent_Variable?.toFixed(2) + "%",
                    MissingPercent_Patient: rec.MissingPercent_Patient?.toFixed(2) + "%",
                }));
                setPhaseDataRows(rows);
            } catch (e) {
                console.error("Error loading phase_missingness_results.json:", e);
                setPhaseDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for patientImpRows (patient_importance_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(patientImportanceSummary)) {
                    setPatientImpRows([]);
                    return;
                }
                const order = { M: 0, R: 1, O: 2 } as any;
                const label = (g: string) => (g === "M" ? "High" : g === "R" ? "Medium" : g === "O" ? "Low" : g);
                const rows = patientImportanceSummary
                    .map((r: any, i: number) =>
                        withTwoPercents({
                            id: `${r.PatientID}-${r.Group}-${i}`,
                            PatientID: r.PatientID,
                            Group: label(r.Group),
                            _code: r.Group,
                            Passed: r.Passed,
                            Failed: r.Failed,
                            Total: r.Total,
                            PassPercent: r.PassPercent,
                        })
                    )
                    .sort((a, b) => (a.PatientID !== b.PatientID ? Number(a.PatientID) - Number(b.PatientID) : order[a._code] - order[b._code]));
                let lastPid: string | null = null;
                rows.forEach((r: any) => {
                    if (r.PatientID !== lastPid) {
                        r.__firstOfPatient = true;
                        lastPid = r.PatientID;
                    }
                });
                setPatientImpRows(rows);
            } catch (e) {
                console.error("patient × importance json:", e);
                setPatientImpRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for phasePatientRows (patient_phase_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                const raw = patientPhaseSummary ?? [];
                if (!Array.isArray(raw)) throw new Error("Not an array");
                const PHASES = ["Diagnosis", "Progression", "Recurrence"];
                const byPid: Record<string, any> = {};
                raw.forEach((r: any) => {
                    const pid = String(r.PatientID);
                    if (!byPid[pid]) byPid[pid] = {};
                    byPid[pid][r.Phase] = r;
                });
                const rows: any[] = [];
                Object.keys(byPid)
                    .sort((a, b) => Number(a) - Number(b))
                    .forEach((pid) => {
                        PHASES.forEach((phase) => {
                            const src = byPid[pid][phase] ?? { Total: 0, Missing: 0 };
                            rows.push(
                                withTwoPercents({
                                    id: `${pid}_${phase}`,
                                    PatientID: pid,
                                    Phase: phase,
                                    Present: src.Total - src.Missing,
                                    Missing: src.Missing,
                                    Total: src.Total,
                                    MissingPercent: src.Total ? ((src.Missing / src.Total) * 100).toFixed(1) + "%" : "0.0%",
                                })
                            );
                        });
                    });
                let lastPid: string | null = null;
                rows.forEach((r) => {
                    if (r.PatientID !== lastPid) {
                        r.__firstOfPatient = true;
                        lastPid = r.PatientID;
                    }
                });
                setPhasePatientRows(rows);
            } catch (e) {
                console.error("Error loading patient_phase_summary_results.json", e);
                setPhasePatientRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for Phase ⇢ Entity grid source (phase_entity_results.json)
    useEffect(() => {
        (async () => {
            try {
                const raw = phaseEntityResults ?? [];
                const byPatient: Record<
                    string,
                    { id: string; Entity: string; Diagnosis: boolean; Progression: boolean; Recurrence: boolean }[]
                > = {};
                (raw as any[]).forEach((r: any) => {
                    const pid = String(r.PatientID ?? r.Entity).replace(/^Patient/, "");
                    if (!byPatient[pid]) byPatient[pid] = [];
                    byPatient[pid].push({
                        id: `${r.Entity}-${r.Phase}`,
                        Entity: r.Entity,
                        Diagnosis: r.Phase === "Diagnosis" ? !!r.Present : false,
                        Progression: r.Phase === "Progression" ? !!r.Present : false,
                        Recurrence: r.Phase === "Recurrence" ? !!r.Present : false,
                    });
                });
                const summary = Object.entries(byPatient).map(([pid, rows]) => {
                    const collapse = (phase: "Diagnosis" | "Progression" | "Recurrence") => {
                        const relevant = rows.filter((r) => r[phase] !== null);
                        if (relevant.length === 0) return null;
                        return relevant.some((r) => r[phase] === true);
                    };
                    return { id: pid, PatientID: pid, Diagnosis: collapse("Diagnosis"), Progression: collapse("Progression"), Recurrence: collapse("Recurrence") };
                });
                patientTimeline.current = byPatient;
                setPhaseRows(summary);
            } catch (e) {
                console.error("Error loading phase_entity_results.json", e);
                setPhaseRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for patient-entity-phase data (patient_entity_phase_results.json)
    useEffect(() => {
        (async () => {
            try {
                const raw = (patientEntityPhaseResults as any[]) ?? [];
                const byPid: Record<string, any[]> = {};
                raw.forEach((rec: any) => {
                    const pid = String(rec.PatientID);
                    if (!byPid[pid]) byPid[pid] = [];
                    let row = byPid[pid].find((x) => x.Entity === rec.Entity);
                    if (!row) {
                        row = { id: `${rec.Entity}-${pid}`, Entity: rec.Entity, Diagnosis: null, Progression: null, Recurrence: null };
                        byPid[pid].push(row);
                    }
                    row[rec.Phase as "Diagnosis" | "Progression" | "Recurrence"] = rec.Complete;
                    row[`Missing${rec.Phase}` as "MissingDiagnosis" | "MissingProgression" | "MissingRecurrence"] = rec.Missing;
                });
                patientTimeline.current = byPid;
                const summary = Object.entries(byPid).map(([pid, rows]) => {
                    const collapse = (phase: "Diagnosis" | "Progression" | "Recurrence"): boolean | "mixed" | null => {
                        const relevant = rows.filter((r: any) => r[phase] !== null);
                        if (relevant.length === 0) return null;
                        const someTrue = relevant.some((r: any) => r[phase] === true);
                        const someFalse = relevant.some((r: any) => r[phase] === false);
                        if (someTrue && someFalse) return "mixed";
                        return someTrue;
                    };
                    return { id: pid, PatientID: pid, Diagnosis: collapse("Diagnosis"), Progression: collapse("Progression"), Recurrence: collapse("Recurrence") };
                });
                setPhaseRows(summary);
            } catch (e) {
                console.error("Error loading patient_entity_phase_results.json", e);
                setPhaseRows([]);
            }
        })();
    }, []);

    // MODIFIED In-Set QC useEffect (in_set_*.json imports)
    useEffect(() => {
        (async () => {
            try {
                const perInstance: any[] = (inSetPerInstance as any) ?? [];
                const perInstanceMapped = perInstance.map((d, i) => ({
                    id: `${d.Variable}-${d.Entity}-${d.EntityInstance ?? i}`,
                    variable: d.Variable,
                    entity: d.Entity,
                    entityInstance: d.EntityInstance ?? "-",
                    passed: d.Failed === false,
                    failed: !!d.Failed,
                    expected: Array.isArray(d.ExpectedSet) ? d.ExpectedSet.join(", ") : "",
                    error: Array.isArray(d.UnexpectedValues) ? d.UnexpectedValues.join(", ") : "",
                }));
                setInSetPerInstanceRows(perInstanceMapped);

                const results: any[] = (inSetResults as any) ?? [];
                setInSetResultsSummary(results);
            } catch (err) {
                console.error("Failed to load In-Set QC data", err);
                setInSetPerInstanceRows([]);
                setInSetResultsSummary([]);
            }
        })();
    }, []);
    // ─────────────────────────────────────────────────────────────


    // const columnsByDatasourceCsv: GridColDef[] = [
    //     { field: "Datasource", headerName: "Datasource", width: 170 },
    //     { field: "Section", headerName: "Section", width: 120 },     // "overall" | "dimension"
    //     { field: "Dimension", headerName: "Dimension", width: 140 }, // Plausibility | Conformance | Completeness | ""
    //     { field: "Scope", headerName: "Scope", width: 140 },         // "checks" | "variables" | "patients"
    //     { field: "Unit", headerName: "Unit", width: 180 },           // kept for compatibility
    //     { field: "Passed", headerName: "Passed", width: 120 },
    //     { field: "Failed", headerName: "Failed", width: 120 },
    //     { field: "Total", headerName: "Total", width: 120 },
    //     { field: "CompletePercent", headerName: "Complete %", width: 140 },
    //     { field: "MissingPercent", headerName: "Missing %", width: 140 },
    // ];


    /* ---------------------------------------------------------------------- */
    /*  Build patient ➜ [{Variable, QC, Value, Entity}] lookup once on mount  */
    /* ---------------------------------------------------------------------- */
    // const patientFailures = React.useRef<Map<string, any[]>>(new Map());

    /** ensure a row always has both MissingPercent & PassPercent */
    // const withTwoPercents = (row: any) => {
    //     if (row.MissingPercent == null && row.PassPercent != null) {
    //         // only “complete/pass” present  → derive missingness
    //         const v = parseFloat(String(row.PassPercent).replace("%", ""));
    //         row.MissingPercent = (100 - v).toFixed(1) + "%";
    //     } else if (row.PassPercent == null && row.MissingPercent != null) {
    //         // only “missing” present → derive completeness
    //         const v = parseFloat(String(row.MissingPercent).replace("%", ""));
    //         row.PassPercent = (100 - v).toFixed(1) + "%";
    //     }
    //     return row;
    // };

    // MODIFIED useEffect for summaryData (dimension_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!dimensionSummary || typeof dimensionSummary !== "object") {
                    setSummaryData([]);
                    return;
                }
                const categoriesInOrder = ["Plausibility", "Conformance", "Completeness", "Total"];
                let idCounter = 1;
                const transformedData = categoriesInOrder
                    .map((categoryName) => {
                        const data = (dimensionSummary as any)[categoryName];
                        return data
                            ? withTwoPercents({
                                id: idCounter++,
                                category: categoryName,
                                Pass: data.Passed,
                                Fail: data.Failed,
                                Amount: data.Total,
                                PassPercent: data.PercentagePass,
                            })
                            : null;
                    })
                    .filter(Boolean) as any[];
                setSummaryData(transformedData);
            } catch (error) {
                console.error("Error loading dimension_summary_results.json:", error);
                setSummaryData([]);
            }
        })();
    }, []); // Empty dependency array to run once on mount

    // MODIFIED useEffect for patientDataRows (patient_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(patientSummary)) throw new Error("Not an array");
                const rows = patientSummary.map((item: any, i: number) =>
                    withTwoPercents({
                        id: item.PatientID || i,
                        PatientID: item.PatientID,
                        isPassed: item.Failed === 0,
                        Pass: item["Number of Passed Tests"],
                        Fail: item.Failed,
                        Amount: item.Total,
                        PassPercent:
                            typeof item.PercentagePass === "number"
                                ? item.PercentagePass.toFixed(2) + "%"
                                : item.PercentagePass,
                    })
                );
                setPatientDataRows(rows);
            } catch (e) {
                console.error("Error loading patient_summary_results.json:", e);
                setPatientDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for qcDataRows (qc_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(qcSummary)) {
                    setQcDataRows([]);
                    return;
                }
                const rows: any[] = [];
                qcSummary.forEach((item: any, idx: number) => {
                    const addRow = (unit: string, passed: number, failed: number, completePct: number, missingPct: number) =>
                        rows.push({
                            id: `${idx}-${unit}`,
                            QC_name: item.QC ?? item.ge_name,
                            Unit: unit,
                            Passed: passed,
                            Failed: failed,
                            CompletePercent: pct(completePct),
                            MissingPercent: pct(missingPct),
                        });
                    addRow("found variables", item.passed_variables, item.failed_variables, item.complete_variables_percentage, item.missing_variables_percentage);
                    addRow("full dm variables", item.full_dm_passed_variables, item.full_dm_failed_variables, item.full_dm_complete_variables_percentage, item.full_dm_missing_variables_percentage);
                    addRow("patients", item.passed_patients, item.failed_patients, item.complete_patients_percentage, item.missing_patients_percentage);
                    addRow("performed quality checks", item.passed_quality_checks, item.failed_quality_checks, item.quality_checks_percentage_pass, item.quality_checks_percentage_fail);
                });
                setQcDataRows(rows);
            } catch (error) {
                console.error("Error loading qc_summary_results.json:", error);
                setQcDataRows([]);
            }
        })();
    }, []); // Empty dependency array to run once on mount

    // MODIFIED useEffect for variableDataRows (variable_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(variableSummary)) {
                    setVariableDataRows([]);
                    return;
                }
                const rows = variableSummary.map((item: any, idx: number) =>
                    withTwoPercents({
                        id: item.Variable ?? idx,
                        Variable: item.Variable,
                        isPassed: item.Failed === 0,
                        Pass: item.Passed,
                        Fail: item.Failed,
                        Amount: item.Total,
                        PassPercent:
                            typeof item.PercentagePass === "number"
                                ? item.PercentagePass.toFixed(2) + "%"
                                : item.PercentagePass,
                    })
                );
                setVariableDataRows(rows);
            } catch (e) {
                console.error("Error loading variable_summary_results.json:", e);
                setVariableDataRows([]);
            }
        })();
    }, []);

    /* ---- Var × Patient matrix (CSV) --------------------------------------- */
    useEffect(() => {
        (async () => {
            try {
                const url = "/data/results/variable_patient_missing_matrix.csv";
                const res = await fetch(url, { cache: "no-store" });
                if (!res.ok) {
                    if (res.status !== 404) {
                        console.error("Var×Patient matrix fetch error:", res.status, res.statusText);
                    }
                    setMatrixCols([]);
                    setMatrixRows([]);
                    return;
                }

                const csvText = await res.text();

                // Sniff delimiter (comma vs semicolon)
                const count = (ch: string) => (csvText.match(new RegExp(`\\${ch}`, "g")) || []).length;
                const FS = count(";") > count(",") ? ";" : ",";

                const wb = read(csvText, { type: "string", raw: true, FS });
                const ws = wb.Sheets[wb.SheetNames[0]];
                const json = utils.sheet_to_json(ws, { header: 1 }) as string[][];

                if (json.length < 2) {
                    setMatrixCols([]);
                    setMatrixRows([]);
                    return;
                }

                const [header, ...body] = json;

                const cols = header.map((h) => ({
                    field: h,
                    headerName: h,
                    width: 120,
                    renderCell:
                        h === "Variable" || h === "Importance"
                            ? undefined
                            : (p: any) => (p.value === "Missing" ? <CancelIcon color="error" /> : p.value ?? ""),
                }));
                setMatrixCols(cols);

                const rows = body.map((arr, idx) => {
                    const row: any = { id: idx };
                    header.forEach((h, i) => {
                        row[h] = i < arr.length && arr[i] !== undefined ? arr[i] : "";
                    });
                    return row;
                });
                setMatrixRows(rows);
            } catch (e) {
                console.error("Var×Patient matrix load error:", e);
                setMatrixCols([]);
                setMatrixRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for datasourceDataRows (datasource_missingness_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(datasourceMissingness)) {
                    setDatasourceDataRows([]);
                    return;
                }
                const normalRows: any[] = [];
                const unknownRows: any[] = [];
                const maybePush = (row: any, ds: string) =>
                    (ds.toLowerCase() === "unknown_source" ? unknownRows : normalRows).push(row);

                datasourceMissingness.forEach((item: any, idx: number) => {
                    const addRow = (unit: string, passed: number, failed: number, completePct: number, missingPct: number, cls?: string) =>
                        maybePush(
                            {
                                id: `${item.Datasource}-${unit}-${idx}`,
                                Datasource: item.Datasource,
                                Unit: unit,
                                Passed: passed,
                                Failed: failed,
                                CompletePercent: pct(completePct),
                                MissingPercent: pct(missingPct),
                                __cls: cls,
                            },
                            item.Datasource
                        );
                    addRow("variables", item.passed_variables, item.failed_variables, item.complete_variables_percentage, item.missing_variables_percentage);
                    addRow("patients", item.passed_patients, item.failed_patients, item.complete_patients_percentage, item.missing_patients_percentage);
                    addRow("performed quality checks", item.passed_quality_checks, item.failed_quality_checks, item.quality_checks_percentage_passed, item.quality_checks_percentage_fail);
                    const dims = item.dimensions || {};
                    ["Plausibility", "Conformance", "Completeness"].forEach((dimName) => {
                        const d = dims[dimName];
                        if (!d) return;
                        const vt = Number(d.VarsTotal || 0);
                        const vp = Number(d.VarsPassed || 0);
                        const vf = Number(d.VarsFailed || 0);
                        const compVarPct = vt ? (vp / vt) * 100 : 0;
                        const missVarPct = vt ? (vf / vt) * 100 : 0;
                        addRow(`Variables ${dimName}`, vp, vf, compVarPct, missVarPct, "dim-vars");
                    });
                });
                setDatasourceDataRows([...normalRows, ...unknownRows]);
            } catch (e) {
                console.error("Error loading datasource_missingness_results.json:", e);
                setDatasourceDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for importanceDataRows (importance_group_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(importanceGroupSummary)) throw new Error("Not an array");
                const rows: any[] = [];
                const nice = (g: string) => (g === "M" ? "High" : g === "R" ? "Medium" : g === "O" ? "Low" : g);
                importanceGroupSummary.forEach((rec: any, idx: number) => {
                    const add = (unit: string, passed: number, failed: number, complete: number, missing: number) =>
                        rows.push({
                            id: `${rec.Group}-${unit}-${idx}`,
                            Group: nice(rec.Group),
                            Unit: unit,
                            Passed: passed,
                            Failed: failed,
                            CompletePercent: pct(complete),
                            MissingPercent: pct(missing),
                        });
                    const totVar = rec.total_variables || 0;
                    const passVar = rec.passed_variables || 0;
                    add("found variables", passVar, rec.failed_variables || 0, totVar ? (passVar / totVar) * 100 : 0, totVar ? 100 - (passVar / totVar) * 100 : 0);
                    add("full dm variables", rec.full_dm_passed_variables, rec.full_dm_failed_variables, rec.full_dm_complete_variables_percentage, rec.full_dm_missing_variables_percentage);
                    const totPat = rec.total_patients || 0;
                    const passPat = rec.passed_patients || 0;
                    add("patients", passPat, rec.failed_patients || 0, totPat ? (passPat / totPat) * 100 : 0, totPat ? 100 - (passPat / totPat) * 100 : 0);
                    add("performed quality checks", rec.passed_quality_checks, rec.failed_quality_checks, rec.quality_checks_percentage_passed, rec.quality_checks_percentage_fail);
                });
                setImportanceDataRows(rows);
            } catch (e) {
                console.error("Error loading importance_group_summary_results.json:", e);
                setImportanceDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for datatypeGridRows (datatype_expectation_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(datatypeRows)) throw new Error("Not an array");
                const filtered = datatypeRows.filter((r: any) => String(r?.CurrentDatatype ?? "") !== "Missing");
                const rows = filtered.map((r: any, idx: number) => ({
                    id: r.Variable ?? idx,
                    Variable: r.Variable,
                    ExpectedType: r.ExpectedType,
                    CurrentDatatype: r.CurrentDatatype ?? null,
                    DatatypeCorrect: !!r.DatatypeCorrect,
                    PercentagePass: typeof r.PercentagePass === "number" ? r.PercentagePass.toFixed(2) + "%" : r.PercentagePass ?? "0.00%",
                    PatientsFailed: Array.isArray(r.PatientsFailed) ? r.PatientsFailed : [],
                    Sources: Array.isArray(r.Sources) ? r.Sources : [],
                }));
                setDatatypeRows(rows);
            } catch (e) {
                console.error("Error loading datatype_expectation_results.json", e);
                setDatatypeRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for phaseDataRows (phase_missingness_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(phaseMissingness)) throw new Error("Not an array");
                const rows = phaseMissingness.map((rec: any, idx: number) => ({
                    id: rec.Phase ?? idx,
                    Phase: rec.Phase,
                    MissingPercent_Variable: rec.MissingPercent_Variable?.toFixed(2) + "%",
                    MissingPercent_Patient: rec.MissingPercent_Patient?.toFixed(2) + "%",
                }));
                setPhaseDataRows(rows);
            } catch (e) {
                console.error("Error loading phase_missingness_results.json:", e);
                setPhaseDataRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for patientImpRows (patient_importance_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                if (!Array.isArray(patientImportanceSummary)) {
                    setPatientImpRows([]);
                    return;
                }
                const order = { M: 0, R: 1, O: 2 } as any;
                const label = (g: string) => (g === "M" ? "High" : g === "R" ? "Medium" : g === "O" ? "Low" : g);
                const rows = patientImportanceSummary
                    .map((r: any, i: number) =>
                        withTwoPercents({
                            id: `${r.PatientID}-${r.Group}-${i}`,
                            PatientID: r.PatientID,
                            Group: label(r.Group),
                            _code: r.Group,
                            Passed: r.Passed,
                            Failed: r.Failed,
                            Total: r.Total,
                            PassPercent: r.PassPercent,
                        })
                    )
                    .sort((a, b) => (a.PatientID !== b.PatientID ? Number(a.PatientID) - Number(b.PatientID) : order[a._code] - order[b._code]));
                let lastPid: string | null = null;
                rows.forEach((r: any) => {
                    if (r.PatientID !== lastPid) {
                        r.__firstOfPatient = true;
                        lastPid = r.PatientID;
                    }
                });
                setPatientImpRows(rows);
            } catch (e) {
                console.error("patient × importance json:", e);
                setPatientImpRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for phasePatientRows (patient_phase_summary_results.json)
    useEffect(() => {
        (async () => {
            try {
                const raw = patientPhaseSummary ?? [];
                if (!Array.isArray(raw)) throw new Error("Not an array");
                const PHASES = ["Diagnosis", "Progression", "Recurrence"];
                const byPid: Record<string, any> = {};
                raw.forEach((r: any) => {
                    const pid = String(r.PatientID);
                    if (!byPid[pid]) byPid[pid] = {};
                    byPid[pid][r.Phase] = r;
                });
                const rows: any[] = [];
                Object.keys(byPid)
                    .sort((a, b) => Number(a) - Number(b))
                    .forEach((pid) => {
                        PHASES.forEach((phase) => {
                            const src = byPid[pid][phase] ?? { Total: 0, Missing: 0 };
                            rows.push(
                                withTwoPercents({
                                    id: `${pid}_${phase}`,
                                    PatientID: pid,
                                    Phase: phase,
                                    Present: src.Total - src.Missing,
                                    Missing: src.Missing,
                                    Total: src.Total,
                                    MissingPercent: src.Total ? ((src.Missing / src.Total) * 100).toFixed(1) + "%" : "0.0%",
                                })
                            );
                        });
                    });
                let lastPid: string | null = null;
                rows.forEach((r) => {
                    if (r.PatientID !== lastPid) {
                        r.__firstOfPatient = true;
                        lastPid = r.PatientID;
                    }
                });
                setPhasePatientRows(rows);
            } catch (e) {
                console.error("Error loading patient_phase_summary_results.json", e);
                setPhasePatientRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for Phase ⇢ Entity grid source (phase_entity_results.json)
    useEffect(() => {
        (async () => {
            try {
                const raw = phaseEntityResults ?? [];
                const byPatient: Record<
                    string,
                    { id: string; Entity: string; Diagnosis: boolean; Progression: boolean; Recurrence: boolean }[]
                > = {};
                (raw as any[]).forEach((r: any) => {
                    const pid = String(r.PatientID ?? r.Entity).replace(/^Patient/, "");
                    if (!byPatient[pid]) byPatient[pid] = [];
                    byPatient[pid].push({
                        id: `${r.Entity}-${r.Phase}`,
                        Entity: r.Entity,
                        Diagnosis: r.Phase === "Diagnosis" ? !!r.Present : false,
                        Progression: r.Phase === "Progression" ? !!r.Present : false,
                        Recurrence: r.Phase === "Recurrence" ? !!r.Present : false,
                    });
                });
                const summary = Object.entries(byPatient).map(([pid, rows]) => {
                    const collapse = (phase: "Diagnosis" | "Progression" | "Recurrence") => {
                        const relevant = rows.filter((r) => r[phase] !== null);
                        if (relevant.length === 0) return null;
                        return relevant.some((r) => r[phase] === true);
                    };
                    return { id: pid, PatientID: pid, Diagnosis: collapse("Diagnosis"), Progression: collapse("Progression"), Recurrence: collapse("Recurrence") };
                });
                patientTimeline.current = byPatient;
                setPhaseRows(summary);
            } catch (e) {
                console.error("Error loading phase_entity_results.json", e);
                setPhaseRows([]);
            }
        })();
    }, []);

    // MODIFIED useEffect for patient-entity-phase data (patient_entity_phase_results.json)
    useEffect(() => {
        (async () => {
            try {
                const raw = (patientEntityPhaseResults as any[]) ?? [];
                const byPid: Record<string, any[]> = {};
                raw.forEach((rec: any) => {
                    const pid = String(rec.PatientID);
                    if (!byPid[pid]) byPid[pid] = [];
                    let row = byPid[pid].find((x) => x.Entity === rec.Entity);
                    if (!row) {
                        row = { id: `${rec.Entity}-${pid}`, Entity: rec.Entity, Diagnosis: null, Progression: null, Recurrence: null };
                        byPid[pid].push(row);
                    }
                    row[rec.Phase as "Diagnosis" | "Progression" | "Recurrence"] = rec.Complete;
                    row[`Missing${rec.Phase}` as "MissingDiagnosis" | "MissingProgression" | "MissingRecurrence"] = rec.Missing;
                });
                patientTimeline.current = byPid;
                const summary = Object.entries(byPid).map(([pid, rows]) => {
                    const collapse = (phase: "Diagnosis" | "Progression" | "Recurrence"): boolean | "mixed" | null => {
                        const relevant = rows.filter((r: any) => r[phase] !== null);
                        if (relevant.length === 0) return null;
                        const someTrue = relevant.some((r: any) => r[phase] === true);
                        const someFalse = relevant.some((r: any) => r[phase] === false);
                        if (someTrue && someFalse) return "mixed";
                        return someTrue;
                    };
                    return { id: pid, PatientID: pid, Diagnosis: collapse("Diagnosis"), Progression: collapse("Progression"), Recurrence: collapse("Recurrence") };
                });
                setPhaseRows(summary);
            } catch (e) {
                console.error("Error loading patient_entity_phase_results.json", e);
                setPhaseRows([]);
            }
        })();
    }, []);

    // MODIFIED In-Set QC useEffect (in_set_*.json imports)
    useEffect(() => {
        (async () => {
            try {
                const perInstance: any[] = (inSetPerInstance as any) ?? [];
                const perInstanceMapped = perInstance.map((d, i) => ({
                    id: `${d.Variable}-${d.Entity}-${d.EntityInstance ?? i}`,
                    variable: d.Variable,
                    entity: d.Entity,
                    entityInstance: d.EntityInstance ?? "-",
                    passed: d.Failed === false,
                    failed: !!d.Failed,
                    expected: Array.isArray(d.ExpectedSet) ? d.ExpectedSet.join(", ") : "",
                    error: Array.isArray(d.UnexpectedValues) ? d.UnexpectedValues.join(", ") : "",
                }));
                setInSetPerInstanceRows(perInstanceMapped);

                const results: any[] = (inSetResults as any) ?? [];
                setInSetResultsSummary(results);
            } catch (err) {
                console.error("Failed to load In-Set QC data", err);
                setInSetPerInstanceRows([]);
                setInSetResultsSummary([]);
            }
        })();
    }, []);
    // ─────────────────────────────────────────────────────────────

    const basePctCols = [
        { field: "MissingPercent", headerName: "% Missing", width: 110 },
        { field: "PassPercent", headerName: "% Complete", width: 110 },
    ];

    const columns = [
        { field: "category", headerName: "", width: 150 },
        { field: "Pass", headerName: "Total Pass", width: 150 },
        { field: "Fail", headerName: "Total Fail", width: 150 },
        { field: "Amount", headerName: "Total", width: 150 },
        ...basePctCols,
    ];

    const columnsByQC = [
        { field: "QC_name", headerName: "QC", width: 220 },
        { field: "Unit", headerName: "Unit", width: 180 },
        { field: "Passed", headerName: "Passed", width: 120 },
        { field: "Failed", headerName: "Failed", width: 120 },
        { field: "CompletePercent", headerName: "Complete %", width: 140 },
        { field: "MissingPercent", headerName: "Missing %", width: 140 },
    ];


    const columnsByPatient = [
        { field: "PatientID", headerName: "PID", width: 150 },
        {
            field: "isPassed",
            headerName: "",
            width: 50,
            renderCell: (params: any) => {
                if (params.value) return <CheckCircleIcon color="success" />;
                return <CancelIcon color="error" />;
            },
        },
        { field: "Pass", headerName: "Passed", width: 100 },
        { field: "Fail", headerName: "Failed", width: 100 },
        { field: "Amount", headerName: "Total", width: 100 },
        ...basePctCols,
    ];

    const columnsByVariable = [
        { field: "Variable", headerName: "Variable", width: 200 },
        {
            field: "isPassed",
            headerName: "",
            width: 50,
            renderCell: (params: any) =>
                params.value ? <CheckCircleIcon color="success" /> : <CancelIcon color="error" />,
        },
        { field: "Pass", headerName: "Passed", width: 100 },
        { field: "Fail", headerName: "Failed", width: 100 },
        { field: "Amount", headerName: "Total", width: 100 },
        ...basePctCols,
    ];

    const columnsByDatasource = [
        { field: "Datasource", headerName: "Datasource", width: 170 },
        { field: "Unit", headerName: "Unit", width: 180 },
        { field: "Passed", headerName: "Passed", width: 120 },
        { field: "Failed", headerName: "Failed", width: 120 },
        { field: "CompletePercent", headerName: "Complete %", width: 140 },
        { field: "MissingPercent", headerName: "Missing %", width: 140 },
    ];

    const columnsDatatype = [ // ← NEW columns
        { field: "Variable", headerName: "Variable", width: 230 },
        {
            field: "DatatypeCorrect",
            headerName: "",
            width: 50,
            renderCell: (p: any) => p.value ? <CheckCircleIcon color="success" /> : <CancelIcon color="error" />
        },
        { field: "ExpectedType", headerName: "Expected", width: 120 },
        { field: "CurrentDatatype", headerName: "Current", width: 120 },
        { field: "PercentagePass", headerName: "% Pass", width: 90 },
    ];

    const columnsByImportance = [
        { field: "Group", headerName: "Group", width: 150 },
        { field: "Unit", headerName: "Unit", width: 180 },
        { field: "Passed", headerName: "Passed", width: 120 },
        { field: "Failed", headerName: "Failed", width: 120 },
        { field: "CompletePercent", headerName: "Complete %", width: 140 },
        { field: "MissingPercent", headerName: "Missing %", width: 140 },
    ];

    const columnsByPhase = [
        { field: "Phase", headerName: "Phase", width: 140 },
        { field: "MissingPercent_Variable", headerName: "% Missing (Var)", width: 170 },
        { field: "MissingPercent_Patient", headerName: "% Missing (Pat)", width: 170 },
    ];


    const columnsByImpPat = [
        { field: "PatientID", headerName: "PID", width: 110 },
        { field: "Group", headerName: "Group", width: 110 },
        { field: "Passed", headerName: "Passed", width: 100 },
        { field: "Failed", headerName: "Failed", width: 100 },
        { field: "Total", headerName: "Total", width: 100 },
        ...basePctCols,];

    const columnsPhasePatient = [
        { field: "PatientID", headerName: "PID", width: 120 },
        { field: "Phase", headerName: "Phase", width: 140 },
        { field: "Present", headerName: "Present", width: 110 },
        { field: "Missing", headerName: "Missing", width: 110 },
        { field: "Total", headerName: "Total", width: 110 },
        ...basePctCols,
    ];

    const columnsPhaseEntity = [
        { field: "Entity", headerName: "Entity", width: 200 },
        ...["Diagnosis", "Progression", "Recurrence"].map(phase => ({
            field: phase,
            headerName: phase[0],                // shows D / P / R
            width: 80,
            sortable: false,
            renderCell: (p: any) =>
                p.value
                    ? <CheckCircleIcon color="success" />
                    : <CancelIcon color="error" />,
        })),
    ];


    const dividerStyle = {
        "& .patient-divider": {
            borderTop: "2px solid",
            borderColor: "neutral.outlinedBorder",
        },
    };

    // NEW: popup state
    const [popupOpen, setPopupOpen] = React.useState(false);
    const [popupTitle, setPopupTitle] = React.useState<string>("");
    const [popupPatients, setPopupPatients] = React.useState<string[]>([]);
    // NEW: content for popup (used by row click handlers)
    const [popupContent, setPopupContent] = React.useState<React.ReactNode>(null);

    // NEW: helper to open popup only when there are patient ids
    const openPatientsPopup = (title: string, ids?: string[] | null) => {
        const list = (ids || []).filter(Boolean);
        if (list.length > 0) {
            setPopupTitle(title);
            setPopupPatients(list as string[]);
            setPopupOpen(true);
        }
    };

    // NEW: handlers to be passed to your tables
    const handleDatatypeRowClick = useCallback((row: DatatypeRow) => {
        const failedList = (row.PatientsFailed || []).filter(Boolean);
        setPopupTitle(`Datatype – ${row.Variable}`);
        setPopupContent(
            <div>
                <div><strong>Expected:</strong> {row.ExpectedType ?? "-"}</div>
                <div><strong>Current:</strong> {row.CurrentDatatype ?? "-"}</div>
                <div><strong>DatatypeCorrect:</strong> {String(row.DatatypeCorrect ?? false)}</div>
                {row.Sources?.length ? (
                    <div><strong>Sources:</strong> {row.Sources.join(", ")}</div>
                ) : null}
                {failedList.length ? (
                    <div style={{ marginTop: 8 }}>
                        <strong>Patients Failed:</strong> {failedList.join(", ")}
                    </div>
                ) : null}
            </div>
        );
        setPopupOpen(true);
    }, []);

    const handleInSetRowClick = useCallback((row: any) => {
        // normalise fields from the mapped grid row
        const titleVar = row.variable ?? row.Variable ?? "-";
        const entity = row.entity ?? row.Entity ?? "-";
        const entityInstance = row.entityInstance ?? row.EntityInstance ?? "-";
        const failed = typeof row.failed === "boolean"
            ? row.failed
            : (row.passed === false || row.Failed === true);
        const expected = row.expected ?? (Array.isArray(row.ExpectedSet) ? row.ExpectedSet.join(", ") : "-");
        // Only show observed values when failed; otherwise empty
        const observed = failed
            ? (row.error ?? (Array.isArray(row.UnexpectedValues) ? row.UnexpectedValues.join(", ") : ""))
            : "";

        // NEW: collect all failed EntityInstance IDs for same Variable+Entity
        const failedEntityIds = Array.from(new Set(
            (inSetPerInstanceRows || [])
                .filter((r) => (r.variable ?? r.Variable) === titleVar
                    && (r.entity ?? r.Entity) === entity
                    && (r.failed === true || r.passed === false || r.Failed === true))
                .map((r) => (r.entityInstance ?? r.EntityInstance))
                .filter(Boolean)
        ));

        setPopupTitle(`In-set – ${titleVar}`);
        setPopupContent(
            <div>
                <div><strong>Entity:</strong> {entity}</div>
                <div><strong>EntityInstance:</strong> {entityInstance}</div>
                <div><strong>Failed:</strong> {String(failed)}</div>
                <div><strong>Expected Set:</strong> {expected || "-"}</div>
                <div><strong>Observed Values:</strong> {observed}</div>
                {/* NEW: list of failed entity ids for this Variable+Entity */}
                <div><strong>Failed Entity IDs:</strong> {failedEntityIds.join(", ")}</div>
            </div>
        );
        setPopupOpen(true);
    }, [inSetPerInstanceRows]);

    // Memoised small slices to render a minimal list demo (you can remove this UI and just reuse handlers)
    const datatypePreview = useMemo(
        () => (Array.isArray(datatypeRows) ? (datatypeRows as DatatypeRow[]).slice(0, 10) : []),
        []
    );
    const inSetPreview = useMemo(
        () => (Array.isArray(inSetRows) ? (inSetRows as InSetRow[]).slice(0, 10) : []),
        []
    );

    return (
        <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
            <Stack spacing={2} sx={{ flex: 1, overflow: "auto" }}>
                {/* <Navbar /> */}
                <Stack
                    spacing={2}
                    sx={{ alignItems: "center", justifyContent: "center", flexGrow: 1 }}
                >
                    <Typography level="h2">The report about your data:</Typography>

                    {/* <Avatar
            sx={{ width: "8rem", height: "8rem", backgroundColor: "yellow" }}
          >
            <WorkspacePremiumIcon sx={{ color: "white", fontSize: 80 }} />
          </Avatar>
          <Typography>Congratulations, your data seems fantastic!</Typography> */}

                    <Tabs
                        variant="outlined"
                        aria-label="Pricing plan"
                        defaultValue={0}
                        value={activeTab}
                        onChange={(_, v) => setActiveTab(Number(v))}
                        sx={{
                            width: "auto",
                            borderRadius: "lg",
                            boxShadow: "sm",
                            overflow: "auto",
                        }}
                    >
                        <TabList
                            disableUnderline
                            tabFlex={1}
                            sx={{
                                [`& .${tabClasses.root}`]: {
                                    fontSize: "sm",
                                    fontWeight: "lg",
                                    [`&[aria-selected="true"]`]: {
                                        color: "primary.500",
                                        bgcolor: "background.surface",
                                    },
                                    [`&.${tabClasses.focusVisible}`]: {
                                        outlineOffset: "-4px",
                                    },
                                },
                            }}
                        >
                            {/* <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                Summary
              </Tab> */}
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                                By QC
                            </Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                                Var × Patient

                            </Tab>

                            {/* <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                By patient ID
              </Tab> */}
                            {/* <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                By Variable
              </Tab> */}
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                                By datasource
                            </Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                                Datatypes
                            </Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                                By phase
                            </Tab>
                            {/* <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                Importance ⇢ Patient
              </Tab> */}
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                                Phase ⇢ Patient
                            </Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>
                                In-Set QC
                            </Tab>
                        </TabList>
                        {/* <TabPanel value={0}>
              <div style={{ height: "auto", width: "auto" }}>
                <DataGrid rows={summaryData} columns={columns} />
              </div>
            </TabPanel> */}
                        <TabPanel value={0}>
                            <div style={{ height: "auto", width: "auto" }}>
                                <DataGrid rows={qcDataRows} columns={columnsByQC} />
                            </div>
                        </TabPanel>
                        <TabPanel value={1}>
                            <div style={{ height: "70vh", width: "100%" }}>
                                <DataGrid
                                    rows={matrixRows}
                                    columns={matrixCols}
                                    density="compact"
                                    sx={{
                                        "& .MuiDataGrid-cell": { py: 0.5 },
                                    }}
                                />
                            </div>
                        </TabPanel>

                        {/* <TabPanel value={2}>
              <Stack spacing={1}>
                <Typography level="body-sm" sx={{ color: "text.tertiary" }}>
                  👉 Click a patient row to inspect the failed quality checks
                </Typography>

                <div style={{ height: "auto", width: "100%" }}>
                  <DataGrid
                    rows={patientDataRows}
                    columns={columnsByPatient}
                    onRowClick={(params) => {
                      const pid = String(params.row.PatientID);
                                    setFailModalPatient(pid);
                                    setFailModalRows(patientFailures.current.get(pid) ?? []);
                                    setFailModalOpen(true);
                    }}
                  />
                </div>
                            </Stack>
            </TabPanel> */}


                        {/* <TabPanel value={3}>
              <div style={{ height: "auto", width: "100%" }}>
                <DataGrid rows={variableDataRows} columns={columnsByVariable} />
              </div>
            </TabPanel> */}
                        <TabPanel value={2}>
                            <div style={{ height: "auto", width: "100%" }}>
                                <DataGrid
                                    rows={datasourceDataRows}
                                    columns={columnsByDatasource}
                                    getRowClassName={(params) => String(params.row.__cls ?? "")}
                                    sx={{
                                        // subtle background for dimension blocks
                                        "& .dim-checks": {
                                            bgcolor: "rgba(25, 118, 210, 0.08)",   // bluish tint
                                            fontWeight: 600,
                                        },
                                        "& .dim-checks > .MuiDataGrid-cell": {
                                            borderTopWidth: 2,
                                            borderTopStyle: "solid",
                                            borderTopColor: "divider",
                                        },
                                        "& .dim-vars": {
                                            bgcolor: "rgba(25, 118, 210, 0.04)",
                                        },
                                    }}
                                />
                            </div>
                        </TabPanel>
                        {/* <TabPanel value={3}>
              <div style={{ height: "auto", width: "100%" }}>
                <DataGrid rows={importanceDataRows} columns={columnsByImportance} />
              </div>
            </TabPanel> */}
                        <TabPanel value={3}> {/* NEW Datatypes tab panel */}
                            <div style={{ height: "auto", width: "100%" }}>
                                <DataGrid
                                    rows={datatypeGridRows}
                                    columns={columnsDatatype}
                                    onRowClick={({ row }) => handleDatatypeRowClick(row)}
                                />
                            </div>
                        </TabPanel>
                        <TabPanel value={4}>
                            <div style={{ height: "auto", width: "100%" }}>
                                <DataGrid
                                    rows={phaseRows}
                                    columns={phaseColumns}
                                    onRowClick={(params) => {
                                        const pid = String(params.row.PatientID);
                                        setTimelinePID(pid);
                                        setTimelineRows(patientTimeline.current[pid] ?? []);
                                        setTimelineOpen(true);
                                    }}
                                />
                            </div>
                        </TabPanel>
                        <TabPanel value={5}>
                            <div style={{ height: "auto", width: "100%" }}>
                                <DataGrid
                                    rows={phasePatientRows}
                                    columns={columnsPhasePatient}
                                    getRowClassName={(params) => {
                                        const parity = Number(params.row.PatientID) % 2 === 0 ? "pid-even" : "pid-odd";
                                        const divider = params.row.__firstOfPatient ? "patient-divider" : "";
                                        return `${parity} ${divider}`.trim();
                                    }}
                                    sx={{
                                        "& .pid-even": { bgcolor: "#f5f5f5" },
                                        "& .pid-odd": { bgcolor: "background.surface" },
                                        "& .patient-divider > .MuiDataGrid-cell": {
                                            borderTopWidth: 2,
                                            borderTopStyle: "solid",
                                            borderTopColor: "divider",
                                        },
                                    }}
                                />
                            </div>
                        </TabPanel>
                        <TabPanel value={6}>
                            <div style={{ height: "auto", width: "100%" }}>
                                <DataGrid
                                    rows={inSetPerInstanceRows}
                                    columns={inSetColumns}
                                    density="compact"
                                    pageSizeOptions={[10, 25, 50]}
                                    autoHeight
                                    onRowClick={({ row }) => handleInSetRowClick(row)}
                                />
                            </div>
                        </TabPanel>
                    </Tabs>
                </Stack>
                {/* <Divider sx={{ marginTop: 0, marginBottom: 3 }} /> */}

            </Stack>
            <PatientTimeline
                open={timelineOpen}
                onClose={() => setTimelineOpen(false)}
                patientId={timelinePID}
                rows={timelineRows}
            />

            {/* NEW: details popup (Joy UI style) */}

        </Box >
    );
}
