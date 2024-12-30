"use server";

import { db } from "../db";

// Get one group
export const verifyToken = async (token: string, id: string) => {
    
    if (!token || token === "" || !id || id === "") {
        return { error: "Nederīgi dati" };
    }
    
    const verifyToken = await db.verificationToken.findUnique({
        where: { token: token, id: id },
    });

    if (!verifyToken) {
        return { error: "Nederīgs tokens!" };
    }

    if (verifyToken.expires < new Date()) {
        return { error: "Tokena derīguma termiņš ir beidzies :(" };
    }

    const user = await db.user.findUnique({
        where: { id: verifyToken.identifier },
    });

    if (!user) {
        return { error: "Lietotājs nav atrasts :(" };
    }

    if (user.emailVerified) {
        return { error: "E-pasts jau ir apstiprināts!" };
    }

    await db.user.update({
        where: { id: user.id },
        data: { emailVerified: new Date() },
    });

    return verifyToken;
};
