import Button from "@mui/joy/Button";
import ButtonGroup from "@mui/joy/ButtonGroup";
import Stack from "@mui/joy/Stack";
import * as React from "react";

import { Link as RouterLink } from "react-router-dom";

export default function Navbar() {
    return (
        <Stack
            spacing={1}
            direction="row"
            flexWrap="wrap"
            sx={{
                width: "100%",
                paddingLeft: 1,
                bgcolor: "primary.main",
            }}
            useFlexGap
        >
            <ButtonGroup aria-label="button group" variant="solid" color="primary">
                <Button component={RouterLink} to="/home">
                    Home
                </Button>
                <Button>Two</Button>
                <Button>Three</Button>
            </ButtonGroup>
        </Stack>
    );
}
