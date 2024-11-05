"use client";

import { useParams, useRouter } from "next/navigation";
import GenericLayout from "../../../components/generic-page-layout"
import NewObjectForm from "../new-object/page"
import ResidenceObjectForm from "../../../components/masonry-table/ResidenceObjectForm";
import PageHeader from "../../../components/generic-page-layout/page-header/PageHeader";
import { useEffect, useState } from "react";
import { getGroup } from "../../../../actions/group";
import { getObject } from "../../../../actions/groupObjects";

const residenceObjectForm = () => {
    const router = useRouter();
    const params = useParams();

    const group_id = Array.isArray(params.group_id)
        ? params.group_id[0]
        : params.group_id;

    const object_id = Array.isArray(params.object_id) ? params.object_id[0] : params.object_id;

    const [group, setGroup] = useState<any>(null);
    const [residence, setResidence] = useState<any>(null);


    useEffect(() => {
        getGroup(group_id).then((group) => {
            setGroup(group);
        }
        );
        getObject(object_id).then((residence) => {
            setResidence(residence);
        }
        );

    }, []);

 
    return (
        <GenericLayout>
            <PageHeader
          title={group ? group.name : "Mana grupa"}
          breadcrumbItems={[
            { label: "Manas grupas", path: "/groups" },
            {
              label: group ? group.name : "Mana grupa",
              path: `/groups/${group_id}`,
            },
            {
                label: residence ? residence.name : "Mana dzīvesvieta",
                path: `/groups/${group_id}/${object_id}`,
            },
          ]}
        />
            <ResidenceObjectForm />
        </GenericLayout>
    )
}

export default residenceObjectForm;
