
import { verifyToken } from "../../../../actions/verify_email";

import ReturnButton from "./returnButton";

const VerifyPage = async ({
    searchParams,
}: {
    searchParams: { token: string, id: string };
}) => {
    const token = searchParams?.token;

    const id = searchParams?.id;

    if (!token || token === "" || !id || id === "") {
        return (
            <div style={{ textAlign: "center", padding: "2rem" }}>
                <p style={{ color: "red" }}>No token provided for verification.</p>
                <button
                    style={{
                        padding: "0.5rem 1rem",
                        marginTop: "1rem",
                        backgroundColor: "#0070f3",
                        color: "white",
                        border: "none",
                        cursor: "pointer",
                    }}
                >
                    Return to Home
                </button>
            </div>
        );
    }

    const verifyResponse = await verifyToken(token, id);

    const message = verifyResponse && verifyResponse.error ? verifyResponse.error : verifyResponse.message;

    console.log("verifyResponse", verifyResponse);

    return (
        <div style={{ 
            display: "flex", 
            justifyContent: "center", 
            backgroundColor: "#323738",
            width: "100vw",
            height: "100vh",
            alignItems: "center",
            flexDirection: "column",
            boxShadow: "0 4px 8px 0 rgba(0, 0, 0, 0.2)",
            }}>
                <span
                    style={{
                        color: "white",
                        fontSize: "1.8rem",
                        fontWeight: "500",
                        textAlign: "center",
                        margin: "2rem",
                    }}
                >
                    {message}
                </span>
            <div style={{ height: "1px", width: "80%", backgroundColor: "#e0e0e0", margin: "1rem 0" }}></div>
            <ReturnButton text="Atgriezties uz sākumlapu" />
        </div>
        
    );
};

export default VerifyPage;
