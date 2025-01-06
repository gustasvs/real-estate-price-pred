import { NextResponse } from "next/server";
import minioClient from "../minioClient";


export async function GET(req: Request) {
    
    console.log("Request URL:", req);

    const url = new URL(req.url);
    
    try {
        // const fileName = req.nextUrl.searchParams.get('fileName');
        // const bucketName = req.nextUrl.searchParams.get('bucketName');

        
        const fileName = url.searchParams.get('fileName');
        const bucketName = url.searchParams.get('bucketName');
        

        if (!fileName || !bucketName) {
            return new NextResponse('Missing fileName or bucketName', { status: 400 });
        }

        const preSignedUrl = await minioClient.presignedGetObject(bucketName, fileName, 24 * 60 * 60);

        const imageResponse = await fetch(preSignedUrl);
        const arrayBuffer = await imageResponse.arrayBuffer();

        return new Response(arrayBuffer, {
            status: imageResponse.status,
            headers: {
                'Content-Type': imageResponse.headers.get('Content-Type') || 'application/octet-stream'
            }
        });

    } catch (error) {
        console.error('Error fetching image:', error);
        
        return new NextResponse('Error fetching image', { status: 500 });
    }
};
