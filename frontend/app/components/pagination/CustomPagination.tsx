

import { Pagination, PaginationProps } from 'antd';

const CustomPagination = (
    { 
        defaultCurrent,
        defaultPageSize,
        total, 
        onChange, 
        showTotal,
        ...props
    } : {
        total: number,
        onChange: (page: number, pageSize: number) => void,
        showTotal?: (total: number) => string
        [key: string]: any
    }
) => {
    return (
        <Pagination
            locale={{ items_per_page: "lapā" }}
            style={{
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                padding: "1rem",
                borderRadius: "1rem",
                margin: "1rem 0",
                color: "var(--background-light-secondary)",
                outline: "1px solid var(--background-light-main)",
                width: "100%",
            }}
            defaultCurrent={defaultCurrent}
            defaultPageSize={defaultPageSize}
            total={total}
            showSizeChanger
            selectPrefixCls="ant-select"
            simple
            onChange={(page, pageSize) => {
                onChange(page, pageSize);
            }}
            showTotal={showTotal}
            {...props}

        // itemRender={(current, type, originalElement) => {
        //   if (type === "prev") {
        //     return <FaArrowLeft style={{
        //       fontSize: "1.5rem",
        //       color: "var(--background-light-secondary)",
        //     }} />;
        //   }
        //   if (type === "next") {
        //     return <FaArrowLeft style={{
        //       fontSize: "1.5rem",
        //       color: "var(--background-light-secondary)",
        //       transform: "rotate(180deg)",
        //     }} />;
        //   }
        //   return originalElement;
        // }
        // }
        />
    );
}

export default CustomPagination;