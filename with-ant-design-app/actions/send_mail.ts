import nodemailer from 'nodemailer';

async function sendVerificationEmail(email: string, verificationToken: string, id: string, changeEmail: boolean = false) {
    const transporter = nodemailer.createTransport({
        service: 'gmail',
        auth: {
        user: process.env.EMAIL_USER,
        pass: process.env.EMAIL_PASS
        }
    });
  
    try {

    const verificationUrl = `http://localhost:3000/api/auth/verify?token=${verificationToken}&id=${id}`;

    await transporter.sendMail({
        from: process.env.EMAIL_USER,
        to: email,
        subject: changeEmail ? 'Apstipriniet savu jauno e-pasta adresi vietnē "SmartEstate"' : 'Apstipriniet savu e-pasta adresi vietnē "SmartEstate"',
        html: `
            <div style="font-family: Arial, sans-serif; line-height: 1.6;">
                <h2 style="color:rgb(32, 35, 37);">Sveicināti SmartEstate platformā!</h2>
                <p>Lūdzu, apstipriniet ${changeEmail ? "savu jauno" : "savu"} e-pasta adresi, noklikšķinot uz zemāk esošās saites:</p>
                <p>
                    <a href="${verificationUrl}" style="display: inline-block; padding: 10px 20px; margin: 10px 0; font-size: 16px; color: #fff; background-color:rgb(150, 209, 255); text-decoration: none; border-radius: 5px;">
                        ${changeEmail ? "Apstiprināt e-pasta adreses maiņu" : "Apstiprināt e-pasta adresi"}
                    </a>
                </p>
                <p>Ja poga nestrādā, lūdzu, pārkopējiet šo saiti savā pārlūkprogrammā: ${verificationUrl}</p>
                <p style="font-size: 12px; color: #555; margin-top: 20px">Šī saite ir derīga 24 stundas.</p>
                <p style="font-size: 12px; color: #555;">Ja jūs neesat veikuši darbības vietnē "SmartEstate", lūdzu, ignorējiet šo e-pastu.</p>
                <p>Ar cieņu,<br/>SmartEstate</p>
            </div>
        `
    });

    console.log('Email sent');

    } catch (error) {
        console.error('Error sending email:', error);
    } finally {
        transporter.close();
    }
}

export default sendVerificationEmail;