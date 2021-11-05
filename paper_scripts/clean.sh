# super stupid script to do data processing

APPS="bgu bilateral_grid camera_pipe conv_layer hist iir_blur interpolate lens_blur local_laplacian max_filter nl_means stencil_chain unsharp"

for app in $APPS; do
    mv ../apps/${app}/autotuned_samples-2021-11-03-14-27-51/batch_51_0 ../apps/${app}/autotuned_samples-2021-11-02-00-49-04
    rm -r ../apps/${app}/autotuned_samples-2021-11-03-14-27-51/
    bash predict_all.sh ../apps/${app}/autotuned_samples-2021-11-02-00-49-04/batch_51_0 ../apps/${app}/autotuned_samples-2021-11-02-00-49-04/updated.weights ../apps/bgu/autotuned_samples-2021-11-02-00-49-04/predictions_new
done