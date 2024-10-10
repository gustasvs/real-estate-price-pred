"use client";

import React, { useEffect, useState } from 'react';
import { Modal, Table } from 'antd';
import { useRouter } from 'next/navigation';
import GenericLayout from '../../components/generic-page-layout';
import CardTable from '../../components/card-table';

import Layout from 'react-masonry-list';
import MasonryTable from '../../components/masonry-table';

const GroupPage = ({ searchParams }: { searchParams: any }) => {
    const router = useRouter();
    const { group_id } = searchParams;

    const [objects, setObjects] = useState([
      { id: 1, name: "Brīvības iela 12", imageUrl: "/path/to/image1.jpg", height: Math.floor(Math.random() * 200) + 100 },
      { id: 2, name: "Lāčplēša 45", imageUrl: "/path/to/image2.jpg", height: Math.floor(Math.random() * 200) + 100 },
      // { id: 3, name: "Valdemāra iela", imageUrl: "/path/to/image3.jpg", height: Math.floor(Math.random() * 200) + 100 },
      // { id: 4, name: "Pērnavas 6", imageUrl: "/path/to/image4.jpg", height: Math.floor(Math.random() * 200) + 100 },
      // { id: 5, name: "Krasta iela 24, Rīga", imageUrl: "/path/to/image5.jpg", height: Math.floor(Math.random() * 200) + 100 },
      // { id: 6, name: "Āgenskalns", imageUrl: "/path/to/image6.jpg", height: Math.floor(Math.random() * 200) + 100 },
      { id: 7, name: "Tērbatas iela 56", imageUrl: "/path/to/image7.jpg", height: Math.floor(Math.random() * 200) + 100 },
      // { id: 8, name: "Ventspils", imageUrl: "/path/to/image8.jpg", height: Math.floor(Math.random() * 200) + 100 },
      { id: 9, name: "Dzirnavu iela 112, Rīga", imageUrl: "/path/to/image9.jpg", height: Math.floor(Math.random() * 200) + 100 },
      // { id: 10, name: "Vecpilsēta", imageUrl: "/path/to/image10.jpg", height: Math.floor(Math.random() * 200) + 100 },
      { id: 11, name: "Salaspils 32", imageUrl: "/path/to/image11.jpg", height: Math.floor(Math.random() * 200) + 100 },
      // { id: 12, name: "Rīgas iela", imageUrl: "/path/to/image12.jpg", height: Math.floor(Math.random() * 200) + 100 },
      { id: 13, name: "Jelgavas 78", imageUrl: "/path/to/image13.jpg", height: Math.floor(Math.random() * 200) + 100 },
    ]);
    
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
        {/* <CardTable columnCount={4} onCardClick={(id: number) => openObjectsModal(id) }/> */}
        <MasonryTable columnCount={4} onCardClick={(id: number) => openObjectsModal(id) } objects={objects} setObjects={setObjects}/>
      {/* </div> */}
    </GenericLayout>
    )
};

export default GroupPage;