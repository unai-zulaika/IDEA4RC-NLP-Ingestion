import React from "react";
import Modal from "@mui/joy/Modal";
import ModalDialog from "@mui/joy/ModalDialog";
import DialogTitle from "@mui/joy/DialogTitle";
import DialogContent from "@mui/joy/DialogContent";
import Tooltip from "@mui/joy/Tooltip";
import { DataGrid } from "@mui/x-data-grid";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";
import RemoveCircleOutlineIcon from "@mui/icons-material/RemoveCircleOutline";

/**
 * Modal that shows, for one patient, every data‑collecting entity
 * (biopsy, surgery, radiology visit …) and whether that entity has
 * *relevant* data for Diagnosis (D), Progression (P) or Recurrence (R).
 *
 *   ✔︎  — data present for the entity in its *relevant* phase
 *   ✗  — data *absent* in the relevant phase (tooltip lists missing vars)
 *   ○̶  — grey circle/minus → phase *not relevant* for this entity
 */
interface TimelineRow {
    id: string;
    Entity: string;
    Diagnosis: boolean | null;      // null ⇒ not relevant
    Progression: boolean | null;    // null ⇒ not relevant
    Recurrence: boolean | null;     // null ⇒ not relevant
    MissingDiagnosis?: string[];    // only filled when Diagnosis === false
    MissingProgression?: string[];  // "          Progression === false
    MissingRecurrence?: string[];   // "          Recurrence === false
}

interface PatientTimelineProps {
    open: boolean;
    onClose: () => void;
    patientId: string;
    rows: TimelineRow[];
}

const PHASE_KEY: Record<string, keyof TimelineRow> = {
    Diagnosis: "MissingDiagnosis",
    Progression: "MissingProgression",
    Recurrence: "MissingRecurrence",
};

const PatientTimeline: React.FC<PatientTimelineProps> = ({
    open,
    onClose,
    patientId,
    rows,
}) => {
    const columns = React.useMemo(
        () => [
            { field: "Entity", headerName: "Entity", flex: 1, minWidth: 160 },
            ...["Diagnosis", "Progression", "Recurrence"].map((phase) => ({
                field: phase,
                headerName: phase[0], // D / P / R
                width: 90,
                sortable: false,
                renderCell: (p: any) => {
                    const val = p.value as boolean | null;
                    if (val === null || val === undefined) {
                        // Phase not relevant for this entity → grey hollow-minus icon
                        return <RemoveCircleOutlineIcon color="disabled" />;
                    }

                    if (val) {
                        return <CheckCircleIcon color="success" />;
                    }
                    // val === false  ⇒ data missing for this phase
                    const missKey = PHASE_KEY[phase];
                    const missing: string[] | undefined = p.row[missKey];
                    const title = missing && missing.length
                        ? `Missing: ${missing.join(", ")}`
                        : "Missing data";

                    return (
                        <Tooltip title={title} variant="outlined" color="danger">
                            <CancelIcon color="error" />
                        </Tooltip>
                    );
                },
            })),
        ],
        []
    );

    return (
        <Modal open={open} onClose={onClose} className="backdrop-blur-sm">
            <ModalDialog
                size="lg"
                className="rounded-2xl shadow-xl bg-white dark:bg-slate-900"
                sx={{ overflow: "hidden", p: 0 }}
            >
                <DialogTitle className="text-xl font-bold px-6 pt-6 pb-2">
                    Patient {patientId} · timeline
                </DialogTitle>
                <DialogContent className="px-6 pb-6">
                    <div style={{ height: 420, width: "100%" }}>
                        <DataGrid
                            rows={rows}
                            columns={columns}
                            density="compact"
                            hideFooter
                            disableColumnMenu
                            sx={{
                                "& .MuiDataGrid-columnHeader, & .MuiDataGrid-cell": {
                                    outline: "none !important",
                                },
                            }}
                        />
                    </div>
                </DialogContent>
            </ModalDialog>
        </Modal>
    );
};

export default PatientTimeline;