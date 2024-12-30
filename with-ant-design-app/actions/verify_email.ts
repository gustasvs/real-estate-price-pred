"use server";

import { message } from "antd";
import { db } from "../db";

// Get one group
export const verifyToken = async (token: string, id: string) => {
    
    if (!token || token === "" || !id || id === "") {
        return { error: "Nederīgi dati" };
    }
    
    const verifyToken = await db.verificationToken.findUnique({
        where: { token: token, id: id },
    });

    console.log("verifyToken", verifyToken);

    if (!verifyToken) {
        return { error: "Nederīgs tokens" };
    }

    if (verifyToken.expires < new Date()) {
        return { error: "Tokens ir beidzies" };
    }

    const user = await db.user.findUnique({
        where: { id: verifyToken.identifier },
    });

    if (!user) {
        return { error: "Lietotājs nav atrasts :(" };
    }

    if (user.emailVerified !== null) {
        return { error: "E-pasts jau ir apstiprināts!" };
    }

    await db.user.update({
        where: { id: user.id },
        data: { emailVerified: new Date() },
    });

    await db.verificationToken.delete({
        where: { id: verifyToken.id },
    });

    return { message: "E-pasts veiksmīgi apstiprināts!" };
};
