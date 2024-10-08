"use client";

import React, { useEffect, useState } from 'react';
import { Modal, Table } from 'antd';
import { useRouter } from 'next/navigation';
import GenericLayout from '../../components/generic-page-layout';
import CardTable from '../../components/card-table';

const GroupPage = ({ searchParams }: { searchParams: any }) => {
    const router = useRouter();
    const { group_id } = searchParams;

    const [objectModalOpen, setObjectModalOpen] = useState(false);

    const openObjectsModal = (id: number) => {
        console.log(`Open objects for group with id ${id}`);
        setObjectModalOpen(true);
    }

    return (
   <GenericLayout>
      {/* <div style={{ display: "flex", justifyContent: "center" }} className={styles["groups-page"]}> */}
        <Modal
            title="Objects"
            open={objectModalOpen}
            onCancel={() => setObjectModalOpen(false)}
            footer={null}
        />
        <CardTable columnCount={4} onCardClick={(id: number) => openObjectsModal(id) }/>
      {/* </div> */}
    </GenericLayout>
    )
};

export default GroupPage;