"use client";

import { useEffect, useState } from "react";
import { getMyFavoriteObjects } from "../../../../../actions/groupObjects";
import MasonryTable from "../../../masonry-table";

const MyFavouritedObjects = () => {


    interface FavouriteObject {
        id: string;
        name: string;
        address: string;
        area: number;
        description: string;
        favourite: boolean;
        bedroomCount: number;
        bathroomCount: number;
        parkingCount: number;
        price: number;
        predictedPrice: number;
        groupId: string;
        pictures: string[];
        createdAt: Date;
        updatedAt: Date;
    }

    const [myFavouriteObjects, setMyFavouriteObjects] = useState<FavouriteObject[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchFavoriteObjects = async () => {
            try {
                const objects = await getMyFavoriteObjects();
                setMyFavouriteObjects(objects);
            } catch (err) {
                setError(err);
            } finally {
                setLoading(false);
            }
        };

        fetchFavoriteObjects();
    }, []);

  return (
    <>
      <MasonryTable
        columnCount={3}
        objects={myFavouriteObjects}
        onCardFavorite={() => {}}
        createObject={() => {}}
        deleteObject={() => {}}
        updateObject={() => {}}
        loading={false}
        showNavigateToGroup={true}
      />
    </>
  );
}


export default MyFavouritedObjects;