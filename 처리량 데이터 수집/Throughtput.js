    function getAverageThroughput(mediaType, isDynamic) {


        if(tempcontent !=  JSON.stringify(throughputDict.video)){
            console.log('changed! temp :', tempcontent, 'content : ',  JSON.stringify(throughputDict.video))
            totalcontent += "\n" + JSON.stringify(getAverage(true, "video", isDynamic)) + ',' + JSON.stringify(throughputDict.video);
            tempcontent =  JSON.stringify(throughputDict.video);
            contentcount += 1;

            if(contentcount % 40 == 0){
                const file = new File([totalcontent], 'data.txt', { type: 'text/plain'});
        const url = URL.createObjectURL(file);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'throughtput_bbb.txt';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        } else {
            tempcontent = JSON.stringify(throughputDict.video);
        }

            }
        if (tempcontent === undefined){
            tempcontent = JSON.stringify(throughputDict.video);
        }


        return getAverage(true, mediaType, isDynamic);
    }
