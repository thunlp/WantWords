//star
$(document).ready(function(){
    var stepW = 24;
    var starRes = new Array("毫无关系","基本相关","非常匹配");
    var stars = $("#star > li");
    var starResTemp;
    $("#showb").css("width",0);
    stars.each(function(i){
        $(stars[i]).click(function(e){
            var n = i+1;
            $("#showb").css({"width":stepW*n});
            starResTemp = starRes[i];
            $(this).find('a').blur();
            return stopDefault(e);
            return starResTemp;
        });
    });
    stars.each(function(i){
        $(stars[i]).hover(
            function(){
                $(".starRes").text(starRes[i]);
            },
            function(){
                if(starResTemp != null)
                    $(".starRes").text(starResTemp);
                else 
                    $(".starRes").text("");
            }
        );
    });
});
function stopDefault(e){
    if(e && e.preventDefault)
           e.preventDefault();
    else
           window.event.returnValue = false;
    return false;
};
