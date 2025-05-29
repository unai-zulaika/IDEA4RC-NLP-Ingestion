
import React, { useEffect } from "react";
import Box from "@mui/joy/Box";
import Stack from "@mui/joy/Stack";
import Divider from "@mui/joy/Divider";
import Typography from "@mui/joy/Typography";
import WorkspacePremiumIcon from "@mui/icons-material/WorkspacePremium";
import Avatar from "@mui/joy/Avatar";
import { DataGrid } from "@mui/x-data-grid";
import Tabs from "@mui/joy/Tabs";
import TabList from "@mui/joy/TabList";
import Tab, { tabClasses } from "@mui/joy/Tab";
import TabPanel from "@mui/joy/TabPanel";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import CancelIcon from "@mui/icons-material/Cancel";

export default function Results() {
    const [summaryData, setSummaryData] = React.useState([]);
    const [patientDataRows, setPatientDataRows] = React.useState([]);
    const [qcDataRows, setQcDataRows] = React.useState([]);

    useEffect(() => {
        const fetchDimensionData = async () => {
            try {
                const response = await fetch("/dimension_summary_results.json");
                const dimensionSummary = await response.json();

                const categoriesInOrder = ["Plausibility", "Conformance", "Completeness", "Total"];
                let idCounter = 1;

                const transformedData = categoriesInOrder.map(categoryName => {
                    const data = dimensionSummary[categoryName];
                    return {
                        id: idCounter++,
                        category: categoryName,
                        Pass: data?.Passed || 0,
                        Fail: data?.Failed || 0,
                        Amount: data?.Total || 0,
                        PassPercent: data?.PercentagePass || "0.0%",
                    };
                });

                setSummaryData(transformedData);
            } catch (error) {
                console.error("Error loading dimension_summary_results.json:", error);
                setSummaryData([]);
            }
        };

        fetchDimensionData();
    }, []);

    useEffect(() => {
        const fetchPatientData = async () => {
            try {
                const response = await fetch("/patient_summary_results.json");
                const patientSummary = await response.json();

                if (!Array.isArray(patientSummary)) throw new Error("Invalid patient summary data");

                const transformedData = patientSummary.map((item, index) => ({
                    id: item.PatientID || index,
                    PatientID: item.PatientID,
                    isPassed: item.Failed === 0,
                    Pass: item["Number of Passed Tests"],
                    Fail: item.Failed,
                    Amount: item.Total,
                    PassPercent: typeof item["Percentage of pass"] === "number"
                        ? item["Percentage of pass"].toFixed(2) + "%"
                        : "N/A",
                }));

                setPatientDataRows(transformedData);
            } catch (error) {
                console.error("Error loading patient_summary_results.json:", error);
                setPatientDataRows([]);
            }
        };

        fetchPatientData();
    }, []);

    useEffect(() => {
        const fetchQcData = async () => {
            try {
                const response = await fetch("/qc_summary_results.json");
                const qcSummary = await response.json();

                if (!Array.isArray(qcSummary)) throw new Error("Invalid QC summary data");

                const transformedData = qcSummary.map((item, index) => ({
                    id: item.ge_name || index,
                    QC: item.ge_name,
                    isPassed: item.failed_checks === 0,
                    Pass: item.passed_checks,
                    Fail: item.failed_checks,
                    Amount: item.total_checks,
                    PassPercent: typeof item.percentage_pass === "number"
                        ? item.percentage_pass.toFixed(2) + "%"
                        : "N/A",
                }));

                setQcDataRows(transformedData);
            } catch (error) {
                console.error("Error loading qc_summary_results.json:", error);
                setQcDataRows([]);
            }
        };

        fetchQcData();
    }, []);

    const columns = [
        { field: "category", headerName: "", width: 150 },
        { field: "Pass", headerName: "Total Pass", width: 150 },
        { field: "Fail", headerName: "Total Fail", width: 150 },
        { field: "Amount", headerName: "Total", width: 150 },
        { field: "PassPercent", headerName: "Total % Pass", width: 150 },
    ];

    const columnsByQC = [
        { field: "QC", headerName: "QC", width: 150 },
        {
            field: "isPassed",
            headerName: "",
            width: 50,
            renderCell: (params) => {
                return params.value ? <CheckCircleIcon color="success" /> : <CancelIcon color="error" />;
            },
        },
        { field: "Pass", headerName: "Passed", width: 100 },
        { field: "Fail", headerName: "Failed", width: 100 },
        { field: "Amount", headerName: "Total", width: 100 },
        { field: "PassPercent", headerName: "Total % Pass", width: 100 },
    ];

    const columnsByPatient = [
        { field: "PatientID", headerName: "PID", width: 150 },
        {
            field: "isPassed",
            headerName: "",
            width: 50,
            renderCell: (params) => {
                return params.value ? <CheckCircleIcon color="success" /> : <CancelIcon color="error" />;
            },
        },
        { field: "Pass", headerName: "Passed", width: 100 },
        { field: "Fail", headerName: "Failed", width: 100 },
        { field: "Amount", headerName: "Total", width: 100 },
        { field: "PassPercent", headerName: "Total % Pass", width: 100 },
    ];

    return (
        <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
            <Stack spacing={2} sx={{ flex: 1, overflow: "auto" }}>
                <Stack spacing={2} sx={{ alignItems: "center", justifyContent: "center", flexGrow: 1 }}>
                    <Typography level="h2">The report about your data:</Typography>
                    <Avatar sx={{ width: "8rem", height: "8rem", backgroundColor: "yellow" }}>
                        <WorkspacePremiumIcon sx={{ color: "white", fontSize: 80 }} />
                    </Avatar>
                    <Typography>Congratulations, your data seems fantastic!</Typography>

                    <Tabs variant="outlined" defaultValue={0} sx={{ borderRadius: "lg", boxShadow: "sm", overflow: "auto" }}>
                        <TabList disableUnderline tabFlex={1} sx={{ [`& .${tabClasses.root}`]: { fontSize: "sm", fontWeight: "lg" } }}>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>Summary</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>By QC</Tab>
                            <Tab disableIndicator variant="soft" sx={{ flexGrow: 1 }}>By patient ID</Tab>
                        </TabList>
                        <TabPanel value={0}><div style={{ height: "auto", width: "auto" }}><DataGrid rows={summaryData} columns={columns} /></div></TabPanel>
                        <TabPanel value={1}><div style={{ height: "auto", width: "auto" }}><DataGrid rows={qcDataRows} columns={columnsByQC} /></div></TabPanel>
                        <TabPanel value={2}><div style={{ height: "auto", width: "auto" }}><DataGrid rows={patientDataRows} columns={columnsByPatient} /></div></TabPanel>
                    </Tabs>
                </Stack>
                <Divider sx={{ marginTop: 0, marginBottom: 3 }} />
            </Stack>
        </Box>
    );
}
