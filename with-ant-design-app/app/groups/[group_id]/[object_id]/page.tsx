"use client";

import { useParams, useRouter } from "next/navigation";
import GenericLayout from "../../../components/generic-page-layout"
import NewObjectForm from "../new-object/page"
import ResidenceObjectForm from "../../../components/masonry-table/ResidenceObjectForm";

const residenceObjectForm = () => {
    const router = useRouter();
    const params = useParams();

    const group_id = Array.isArray(params.group_id)
        ? params.group_id[0]
        : params.group_id;

    const object_id = Array.isArray(params.object_id)

 
    return (
        <GenericLayout>
            <h1>Object Page</h1>
            <ResidenceObjectForm />
        </GenericLayout>
    )
}

export default residenceObjectForm;
